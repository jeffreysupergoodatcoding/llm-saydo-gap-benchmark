"""Sequence models — SASRec, BERT4Rec (sampled CE), GRU4Rec.

Adapted for binary repeat-purchase prediction: take a customer's article sequence,
encode it, project to a scalar logit for P(repeat purchase in next 30d).

We use a simple yet faithful PyTorch implementation; not aiming to match the
exact SASRec/BERT4Rec from the original papers — what matters is that all three
sequence models share the same fair training protocol and same head.
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SeqDataset(Dataset):
    def __init__(self, sequences: list[list[int]], labels: list[int], max_len: int = 64, pad: int = 0):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len
        self.pad = pad

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        seq = self.sequences[i][-self.max_len:]
        if len(seq) < self.max_len:
            seq = [self.pad] * (self.max_len - len(seq)) + seq
        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(self.labels[i], dtype=torch.float32),
        )


class SASRec(nn.Module):
    def __init__(self, n_items: int, dim: int = 64, n_heads: int = 2, n_layers: int = 2, max_len: int = 64, dropout: float = 0.2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(dim, n_heads, dim_feedforward=dim * 4, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(dim, 1)
        self.max_len = max_len

    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        h = self.item_emb(x) + self.pos_emb(pos)
        mask = (x == 0)
        # causal mask
        causal = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        h = self.encoder(h, mask=causal, src_key_padding_mask=mask)
        # use last non-pad position
        lengths = (x != 0).sum(dim=1).clamp(min=1)
        idx = (lengths - 1).clamp(min=0)
        pooled = h[torch.arange(b, device=x.device), idx]
        logit = self.head(pooled).squeeze(-1)
        return logit


class BERT4Rec(nn.Module):
    def __init__(self, n_items: int, dim: int = 64, n_heads: int = 2, n_layers: int = 2, max_len: int = 64, dropout: float = 0.2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(dim, n_heads, dim_feedforward=dim * 4, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        h = self.item_emb(x) + self.pos_emb(pos)
        mask = (x == 0)
        # bidirectional (no causal mask)
        h = self.encoder(h, src_key_padding_mask=mask)
        # mean pool over non-pad
        valid = (~mask).unsqueeze(-1).float()
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        logit = self.head(pooled).squeeze(-1)
        return logit


class GRU4Rec(nn.Module):
    def __init__(self, n_items: int, dim: int = 64, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.gru = nn.GRU(dim, dim, n_layers, dropout=dropout, batch_first=True)
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        h0 = self.item_emb(x)
        out, _ = self.gru(h0)
        lengths = (x != 0).sum(dim=1).clamp(min=1)
        idx = (lengths - 1).clamp(min=0)
        b = x.shape[0]
        pooled = out[torch.arange(b, device=x.device), idx]
        logit = self.head(pooled).squeeze(-1)
        return logit


def _train_model(model, train_loader, val_loader, *, epochs: int = 5, lr: float = 1e-3, device: str = "cpu", weight: float | None = None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    if weight is not None:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=device, dtype=torch.float32))
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    best_val = float("inf")
    best_state = None
    for ep in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logit = model(x)
            loss = loss_fn(logit, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        model.eval()
        v_total, v_n = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logit = model(x)
                v_total += loss_fn(logit, y).item() * len(y)
                v_n += len(y)
        v_loss = v_total / max(1, v_n)
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def fit_sequence_model(
    model_name: str,
    train_seqs, train_labels,
    val_seqs, val_labels,
    n_items: int,
    seed: int = 42,
    epochs: int = 5,
    max_len: int = 64,
    device: str = "cpu",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if model_name == "sasrec":
        model = SASRec(n_items, dim=64, max_len=max_len)
    elif model_name == "bert4rec":
        model = BERT4Rec(n_items, dim=64, max_len=max_len)
    elif model_name == "gru4rec":
        model = GRU4Rec(n_items, dim=64)
    else:
        raise ValueError(model_name)
    model = model.to(device)
    train_ds = SeqDataset(train_seqs, train_labels, max_len=max_len)
    val_ds = SeqDataset(val_seqs, val_labels, max_len=max_len)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)
    pos_rate = max(np.mean(train_labels), 1e-3)
    weight = (1.0 - pos_rate) / pos_rate
    _train_model(model, train_loader, val_loader, epochs=epochs, device=device, weight=weight)
    return model


def predict_sequence(model, seqs, max_len: int = 64, batch_size: int = 512, device: str = "cpu"):
    model.eval()
    ds = SeqDataset(seqs, [0] * len(seqs), max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logit = model(x)
            p = torch.sigmoid(logit).cpu().numpy()
            out.append(p)
    return np.concatenate(out)
