"""
ANFIS — Adaptive Neuro-Fuzzy Inference System (Takagi-Sugeno Type-1)

SPEED FIXES:
  • n_rules capped at 64 (was 3^12 = 531,441 — the main bottleneck)
  • PSO particles=10, iters=20
  • LSE every 3 epochs
  • Recommended: n_features=8, n_mfs=2 from train_compare.py
"""

import copy
import itertools
import numpy as np
import torch
import torch.nn as nn
import skfuzzy as  fuzz
from sklearn.feature_selection import mutual_info_classif


class GaussianMF(nn.Module):
    def __init__(self, n_inputs: int, n_mfs: int):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.center = nn.Parameter(torch.randn(n_inputs, n_mfs))
        self.log_sigma = nn.Parameter(torch.zeros(n_inputs, n_mfs))

    @property
    def sigma(self):
        return torch.exp(self.log_sigma) + 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        c = self.center.unsqueeze(0)
        s = self.sigma.unsqueeze(0)
        return torch.exp(-0.5 * ((x - c) / s) ** 2)


class ANFIS(nn.Module):
    def __init__(self, n_inputs: int, n_mfs: int = 2, n_rules: int = None):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs

        # ★ CAP RULES AT 64 — prevents the 531k rule explosion
        max_rules = n_rules or min(n_mfs ** n_inputs, 64)
        self.n_rules = max_rules

        self.mf = GaussianMF(n_inputs, n_mfs)

        all_rules = list(itertools.product(range(n_mfs), repeat=n_inputs))
        if len(all_rules) > max_rules:
            rng = np.random.default_rng(42)
            chosen = rng.choice(len(all_rules), size=max_rules, replace=False)
            rule_idx = [all_rules[i] for i in sorted(chosen)]
        else:
            rule_idx = all_rules

        self.register_buffer(
            "rule_indices",
            torch.tensor(rule_idx, dtype=torch.long),
        )
        self.n_rules = len(rule_idx)

        self.consequent = nn.Parameter(
            torch.randn(self.n_rules, n_inputs + 1) * 0.01
        )
        self.register_buffer(
            "active_mask", torch.ones(self.n_rules, dtype=torch.bool)
        )

    def init_from_data(self, X: np.ndarray):
        """
        OPTIMIZED: Uses Fuzzy C-Means (FCM) Clustering instead of 
        simple percentiles to find the natural 'centers' of the voice data.
        """
        n_mfs = self.n_mfs
        with torch.no_grad():
            for i in range(self.n_inputs):
                col = X[:, i]
                
                # --- SOFT COMPUTING IMPROVEMENT: FUZZY C-MEANS ---
                # We tell the computer to find 'n_mfs' (e.g., 2 or 3) natural 
                # clusters in the data for this specific voice feature.
                # cntr = the centers of the clusters (e.g., the 'average' Low and High)
                cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                    col.reshape(1, -1), c=n_mfs, m=2, error=0.005, maxiter=1000
                )
                
                # We sort them so that index 0 is 'Low' and index 1 is 'High'
                centers = np.sort(cntr.flatten())
                self.mf.center.data[i] = torch.tensor(centers, dtype=torch.float32)
                # ------------------------------------------------
                
                # We calculate the 'Spread' (width) of the curves.
                # CRITICAL FIX: Divisor changed from 1.5 to 4.0.
                # With scaled data, spread≈6, so sigma≈6/(4*3)=0.5.
                # This keeps 'Low','Med','High' sharp and DISTINCT —
                # adjacent MFs at distance ~3 apart will have near-zero
                # overlap, giving each rule a unique firing strength
                # instead of all firing at 1.0 and averaging to 0.50.
                spread = max(col.max() - col.min(), 1e-3)
                sigma_val = spread / (4.0 * n_mfs)
                self.mf.log_sigma.data[i] = torch.full(
                    (n_mfs,), float(np.log(sigma_val + 1e-6))
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        mu = self.mf(x)

        firing = torch.ones(batch, self.n_rules, device=x.device)
        for i in range(self.n_inputs):
            mf_sel = self.rule_indices[:, i]
            firing = firing * mu[:, i, mf_sel]

        firing = firing * self.active_mask.float().unsqueeze(0)
        firing_sum = firing.sum(dim=1, keepdim=True) + 1e-8
        w_bar = firing / firing_sum

        x_aug = torch.cat([x, torch.ones(batch, 1, device=x.device)], dim=1)
        rule_out = torch.matmul(x_aug, self.consequent.T)
        output = (w_bar * rule_out).sum(dim=1)
        return torch.sigmoid(output).clamp(1e-7, 1 - 1e-7)

    def hybrid_train(self, X, y, epochs=200, lr=0.005, batch_size=32, verbose=True):
        device = next(self.parameters()).device
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)

        n_pos = float(y.sum())
        n_neg = float(len(y) - n_pos)
        w_pos = len(y) / (2 * max(n_pos, 1))
        w_neg = len(y) / (2 * max(n_neg, 1))
        sample_weights = torch.where(
            y_t == 1,
            torch.tensor(w_pos, device=device),
            torch.tensor(w_neg, device=device),
        )

        optimiser = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=20
        )

        losses = []
        n = X_t.shape[0]

        for epoch in range(epochs):
            if epoch % 3 == 0:
                with torch.no_grad():
                    mu = self.mf(X_t)
                    firing = torch.ones(n, self.n_rules, device=device)
                    for i in range(self.n_inputs):
                        mf_sel = self.rule_indices[:, i]
                        firing = firing * mu[:, i, mf_sel]
                    firing = firing * self.active_mask.float().unsqueeze(0)
                    fs = firing.sum(dim=1, keepdim=True) + 1e-8
                    w_bar = firing / fs
                    x_aug = torch.cat([X_t, torch.ones(n, 1, device=device)], dim=1)
                    A = (w_bar.unsqueeze(-1) * x_aug.unsqueeze(1)).reshape(n, -1)
                    A_w = A * sample_weights.unsqueeze(-1).sqrt()
                    # epsilon prevents log(0) / log(inf) NaN during LSE
                    y_logit = torch.log((y_t + 1e-6) / (1 - y_t + 1e-6))
                    y_logit_w = y_logit * sample_weights.sqrt()
                    try:
                        sol = torch.linalg.lstsq(A_w, y_logit_w.unsqueeze(-1))
                        p = sol.solution.squeeze(-1)
                        self.consequent.data = p.reshape(self.n_rules, self.n_inputs + 1)
                    except Exception:
                        pass

            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start: start + batch_size]
                xb, yb = X_t[idx], y_t[idx]
                wb = sample_weights[idx]
                pred = self.forward(xb)
                loss_per = nn.functional.binary_cross_entropy(pred, yb, reduction="none")
                loss = (loss_per * wb).mean()
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimiser.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            scheduler.step(avg_loss)

            if verbose and (epoch + 1) % 20 == 0:
                print(f"  [ANFIS] Epoch {epoch+1:>4}/{epochs}  Loss={avg_loss:.4f}")

        return losses

    def prune_rules(self, X, threshold=0.002, keep_min=16):
        device = next(self.parameters()).device
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            mu = self.mf(X_t)
            firing = torch.ones(X_t.shape[0], self.n_rules, device=device)
            for i in range(self.n_inputs):
                mf_sel = self.rule_indices[:, i]
                firing = firing * mu[:, i, mf_sel]
            fs = firing.sum(dim=1, keepdim=True) + 1e-8
            w_bar = firing / fs
            avg_firing = w_bar.mean(dim=0)

        sorted_idx = torch.argsort(avg_firing, descending=True)
        mask = avg_firing >= threshold
        if mask.sum() < keep_min:
            mask[:] = False
            mask[sorted_idx[:keep_min]] = True
        self.active_mask.copy_(mask)
        n_active = int(mask.sum().item())
        print(f"  [ANFIS] Pruned rules: {self.n_rules} → {n_active} active")

    def get_rules(self, feature_names=None, top_k=10):
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.n_inputs)]
        labels = ["LOW", "MED", "HIGH"] if self.n_mfs == 3 else (
            ["LOW", "HIGH"] if self.n_mfs == 2 else [f"MF{i}" for i in range(self.n_mfs)]
        )
        device = next(self.parameters()).device
        rules = []
        with torch.no_grad():
            bias_idx = self.n_inputs
            rule_weights = self.consequent[:, bias_idx].abs()
            active_idx = torch.where(self.active_mask)[0]
            if len(active_idx) == 0:
                return ["No active rules."]
            active_weights = rule_weights[active_idx]
            topk = min(top_k, len(active_idx))
            _, topk_idx = torch.topk(active_weights, topk)
            for rank, ti in enumerate(topk_idx):
                ri = active_idx[ti].item()
                mf_combo = self.rule_indices[ri]
                antecedents = []
                for fi, mi in enumerate(mf_combo):
                    label = labels[mi.item()] if mi.item() < len(labels) else f"MF{mi.item()}"
                    antecedents.append(f"{feature_names[fi]} is {label}")
                ante_str = " AND ".join(antecedents)
                x_zeros = torch.zeros(1, self.n_inputs, device=device)
                x_aug = torch.cat([x_zeros, torch.ones(1, 1, device=device)], dim=1)
                out_val = (x_aug @ self.consequent[ri].unsqueeze(-1)).item()
                risk = 1.0 / (1.0 + np.exp(-out_val))
                rules.append(f"Rule {rank+1}: IF {ante_str} THEN risk = {risk:.2f}")
        return rules


def select_top_features(X, y, feature_names, k=8):
    mi = mutual_info_classif(X, y, random_state=42)
    top_idx = np.argsort(mi)[::-1][:k]
    top_idx = np.sort(top_idx)
    selected_names = [feature_names[i] for i in top_idx]
    print(f"  [FS] Selected {k} features by MI: {selected_names}")
    return X[:, top_idx], selected_names, top_idx


def pso_optimize_mf(anfis_model, X_train, y_train, n_particles=10, iters=20, verbose=True):
    import pyswarms as ps

    n_inputs = anfis_model.n_inputs
    n_mfs = anfis_model.n_mfs
    n_center = n_inputs * n_mfs
    n_dim = n_center * 2

    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)

    lower = np.concatenate([np.repeat(x_min, n_mfs), np.full(n_center, -2.0)])
    upper = np.concatenate([np.repeat(x_max, n_mfs), np.full(n_center, 2.0)])

    def fitness(particles):
        costs = np.zeros(len(particles))
        for pi, p in enumerate(particles):
            try:
                centers = p[:n_center].reshape(n_inputs, n_mfs)
                log_sigs = p[n_center:].reshape(n_inputs, n_mfs)
                model = copy.deepcopy(anfis_model)
                model.mf.center.data = torch.tensor(centers, dtype=torch.float32)
                model.mf.log_sigma.data = torch.tensor(log_sigs, dtype=torch.float32)
                model.hybrid_train(X_train, y_train, epochs=5, lr=0.01, verbose=False)
                with torch.no_grad():
                    X_t = torch.tensor(X_train, dtype=torch.float32)
                    y_t = torch.tensor(y_train, dtype=torch.float32)
                    pred = model(X_t).clamp(1e-7, 1 - 1e-7)
                    loss = nn.functional.binary_cross_entropy(pred, y_t)
                costs[pi] = loss.item()
            except Exception:
                costs[pi] = 1e6
        return costs

    optimiser = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=n_dim,
        options={"c1": 1.5, "c2": 1.5, "w": 0.6},
        bounds=(lower, upper),
    )

    if verbose:
        print("  [PSO] Optimising MF parameters...")

    cost, pos = optimiser.optimize(fitness, iters=iters, verbose=verbose)

    best_centers = pos[:n_center].reshape(n_inputs, n_mfs)
    best_log_sigs = pos[n_center:].reshape(n_inputs, n_mfs)
    anfis_model.mf.center.data = torch.tensor(best_centers, dtype=torch.float32)
    anfis_model.mf.log_sigma.data = torch.tensor(best_log_sigs, dtype=torch.float32)

    if verbose:
        print(f"  [PSO] Best cost = {cost:.4f}")
    return anfis_model