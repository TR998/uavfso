from __future__ import annotations

import copy
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from config import Config
from predictors_np import LinearARPredictor, PersistencePredictor, gain_pct, mse


class SNRPredictorNet(nn.Module):
    def __init__(self, hidden=(64, 32), dropout=0.2):
        super().__init__()
        self.l1 = nn.LSTM(1, hidden[0], batch_first=True)
        self.l2 = nn.LSTM(hidden[0], hidden[1], batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden[1], 1)

    def forward(self, x):
        h, _ = self.l1(x.unsqueeze(-1))
        h, _ = self.l2(self.drop(h))
        return self.out(self.drop(h[:, -1])).squeeze(-1)


class LSTMPredictor:
    name = "LSTM"

    def __init__(self, cfg: Config, seed: int = 0, device: str = "cpu"):
        torch.manual_seed(seed)
        self.cfg, self.device, self.seed = cfg, torch.device(device), seed
        self.net = SNRPredictorNet(cfg.lstm_hidden, cfg.lstm_dropout).to(self.device)
        self.history = []

    def fit(self, ds: Dict[str, np.ndarray], epochs: Optional[int] = None, verbose: bool = True,
            max_train: Optional[int] = None) -> Dict:
        c = self.cfg
        epochs = epochs or c.lstm_epochs
        Xtr, ytr = ds["X_train"], ds["y_train"]
        if max_train:
            Xtr, ytr = Xtr[:max_train], ytr[:max_train]
        Xtr = torch.as_tensor(Xtr, device=self.device); ytr = torch.as_tensor(ytr, device=self.device)
        Xva = torch.as_tensor(ds["X_val"], device=self.device); yva = torch.as_tensor(ds["y_val"], device=self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=c.lstm_lr)
        lossf = nn.MSELoss()
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        best, best_state, bad = float("inf"), None, 0
        t0 = time.time()
        for ep in range(1, epochs + 1):
            self.net.train()
            perm = torch.randperm(len(Xtr), generator=g).to(self.device)
            tot = 0.0
            for i in range(0, len(Xtr), c.lstm_batch):
                b = perm[i:i + c.lstm_batch]
                opt.zero_grad()
                loss = lossf(self.net(Xtr[b]), ytr[b])
                loss.backward(); opt.step()
                tot += loss.item() * len(b)
            tr_loss = tot / len(Xtr)
            va_loss = self._eval_mse(Xva, yva)
            self.history.append(dict(epoch=ep, train_mse=tr_loss, val_mse=va_loss))
            if verbose:
                print(f"  epoch {ep:3d}  train MSE {tr_loss:.5f}  val MSE {va_loss:.5f}  ({time.time()-t0:.0f}s)", flush=True)
            if va_loss < best - 1e-7:
                best, best_state, bad = va_loss, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= c.lstm_patience:
                    if verbose:
                        print(f"  early stopping at epoch {ep}")
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return dict(best_val_mse=best, epochs_run=len(self.history), seconds=time.time() - t0)

    @torch.no_grad()
    def _eval_mse(self, X, y) -> float:
        self.net.eval()
        out = []
        for i in range(0, len(X), 4096):
            out.append(self.net(X[i:i + 4096]))
        return float(((torch.cat(out) - y) ** 2).mean())

    @torch.no_grad()
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        X = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
        out = [self.net(X[i:i + 4096]) for i in range(0, len(X), 4096)]
        return np.clip(torch.cat(out).cpu().numpy().astype(np.float64), 0.0, 1.0)

    def predict(self, hist_norm: np.ndarray) -> float:
        return float(self.predict_batch(np.asarray(hist_norm)[None, :])[0])

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        return self

    def n_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())


def compare_predictors(ds: Dict[str, np.ndarray], lstm: LSTMPredictor) -> Dict[str, float]:
    Xte, yte = ds["X_test"], ds["y_test"]
    ar = LinearARPredictor().fit(ds["X_train"], ds["y_train"])
    res = dict(mse_mean=mse(np.full_like(yte, ds["y_train"].mean()), yte),
               mse_persistence=mse(PersistencePredictor().predict_batch(Xte), yte),
               mse_linear_AR=mse(ar.predict_batch(Xte), yte),
               mse_lstm=mse(lstm.predict_batch(Xte), yte))
    res["lstm_gain_vs_persistence_pct"] = gain_pct(res["mse_lstm"], res["mse_persistence"])
    res["lstm_gain_vs_linear_pct"] = gain_pct(res["mse_lstm"], res["mse_linear_AR"])
    return res
