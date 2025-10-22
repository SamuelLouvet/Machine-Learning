import argparse
import sqlite3
import csv
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from net import Net
from load_data import create_data


def parse_args():
    p = argparse.ArgumentParser(description='Summarize metrics from SQLite DB and export CSV, with optional F1/AUC')
    p.add_argument('--db', default='./artifacts/metrics.sqlite')
    p.add_argument('--out', default='./artifacts/metrics.csv')
    # Optional: compute F1/AUC by running the model
    p.add_argument('--checkpoint', default='./artifacts/best_model.pth')
    p.add_argument('--train-dir', default='./train_images')
    p.add_argument('--test-dir', default='./test_images')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--valid-size', type=float, default=0.2)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--image-size', type=int, default=32)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def compute_split_metrics(split: str, loader, model, device) -> Optional[Dict[str, float]]:
    model.eval()
    y_true = []
    y_prob = []
    y_pred = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            y_true.append(labels.cpu())
            y_prob.append(probs.cpu())
            y_pred.append(preds.cpu())
    if not y_true:
        return None
    y_true = torch.cat(y_true).numpy()
    y_prob = torch.cat(y_prob).numpy()
    y_pred = torch.cat(y_pred).numpy()
    try:
        f1 = f1_score(y_true, y_pred, average='binary', pos_label=1)
    except Exception:
        f1 = float('nan')
    try:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    except Exception:
        auc = float('nan')
    return {"f1": float(f1), "auc": float(auc)}


def main():
    args = parse_args()
    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    cur.execute('SELECT epoch, split, loss, acc FROM metrics ORDER BY epoch ASC')
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print('No metrics found')
        return

    # Optionally compute F1/AUC for valid/test using the checkpoint
    computed = {}
    try:
        device = torch.device(args.device)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = Net().to(device)
        model.load_state_dict(checkpoint['model_state'])
        train_loader, valid_loader, test_loader, _ = create_data(
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            batch_size=args.batch_size,
            valid_size=args.valid_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
        )
        computed['valid'] = compute_split_metrics('valid', valid_loader, model, device)
        computed['test'] = compute_split_metrics('test', test_loader, model, device)
    except Exception:
        computed = {}

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'split', 'loss', 'acc', 'f1', 'auc'])
        for epoch, split, loss, acc in rows:
            f1v = ''
            aucv = ''
            if split in computed and computed[split] is not None:
                f1v = computed[split]['f1']
                aucv = computed[split]['auc']
            writer.writerow([epoch, split, loss, acc, f1v, aucv])
    print(f'Wrote {len(rows)} rows to {args.out} (with F1/AUC when available)')


if __name__ == '__main__':
    main()


