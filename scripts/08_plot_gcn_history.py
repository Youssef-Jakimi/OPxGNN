#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple utility to plot train/val loss and F1 curves from
`data/processed/gcn_training_history.json` and save PNG.
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / 'data' / 'processed' / 'gcn_training_history.json'
OUT = ROOT / 'data' / 'processed' / 'gcn_learning_curves.png'

if not HISTORY.exists():
    print(f'Historique introuvable: {HISTORY}')
    raise SystemExit(1)

with open(HISTORY, 'r', encoding='utf-8') as f:
    hist = json.load(f)

epochs = [e['epoch'] for e in hist.get('epochs', [])]
train_loss = [e['train'].get('loss', np.nan) for e in hist.get('epochs', [])]
val_loss = [e['val'].get('loss', np.nan) for e in hist.get('epochs', [])]
train_f1 = [e['train'].get('f1', np.nan) for e in hist.get('epochs', [])]
val_f1 = [e['val'].get('f1', np.nan) for e in hist.get('epochs', [])]

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(epochs, train_loss, label='train loss')
plt.plot(epochs, val_loss, label='val loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

plt.subplot(1,2,2)
plt.plot(epochs, train_f1, label='train F1')
plt.plot(epochs, val_f1, label='val F1')
plt.xlabel('Epoch'); plt.ylabel('F1'); plt.legend(); plt.grid(True)

plt.suptitle('GCN training curves')
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(OUT, dpi=150)
print(f'Saved learning curves to {OUT}')
