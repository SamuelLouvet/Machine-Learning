import argparse
import os
import sqlite3
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from net import Net
from load_data import create_data


def parse_args():
    parser = argparse.ArgumentParser(description='Train face detector CNN')
    parser.add_argument('--train-dir', default='./train_images')
    parser.add_argument('--test-dir', default='./test_images')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--valid-size', type=float, default=0.2)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--image-size', type=int, default=32)
    parser.add_argument('--use-imbalanced-sampler', action='store_true')
    parser.add_argument('--class-weighted-loss', action='store_true', help='Use inverse-frequency class weights')
    parser.add_argument('--augment', action='store_true', help='Enable mild data augmentation')
    parser.add_argument('--threshold', type=float, default=None, help='Decision threshold for class 1 probability')
    parser.add_argument('--auto-threshold', action='store_true', help='Optimize threshold on validation set to maximize F1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out-dir', default='./artifacts')
    parser.add_argument('--save-best-only', action='store_true')
    parser.add_argument('--log-sqlite', action='store_true', help='Log metrics to a SQLite DB')
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    return parser.parse_args()


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def evaluate(model, criterion, loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.append(labels.detach().cpu())
            all_probs.append(probs.detach().cpu())
            all_preds.append(predicted.detach().cpu())
    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    labels_np = torch.cat(all_labels).numpy() if all_labels else np.array([])
    probs_np = torch.cat(all_probs).numpy() if all_probs else np.array([])
    preds_np = torch.cat(all_preds).numpy() if all_preds else np.array([])
    return avg_loss, acc, labels_np, probs_np, preds_np


def maybe_init_sqlite(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        'CREATE TABLE IF NOT EXISTS metrics (epoch INTEGER, split TEXT, loss REAL, acc REAL)'
    )
    conn.commit()
    return conn


def log_sqlite(conn, epoch: int, split: str, loss: float, acc: float):
    cur = conn.cursor()
    cur.execute('INSERT INTO metrics(epoch, split, loss, acc) VALUES (?, ?, ?, ?)', (epoch, split, loss, acc))
    conn.commit()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, 'tb')) if args.tensorboard else None
    conn = maybe_init_sqlite(os.path.join(args.out_dir, 'metrics.sqlite')) if args.log_sqlite else None

    train_loader, valid_loader, test_loader, classes = create_data(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        valid_size=args.valid_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_imbalanced_sampler=args.use_imbalanced_sampler,
        seed=args.seed,
        augment=args.augment,
    )

    device = torch.device(args.device)
    model = Net().to(device)
    # Optionally compute class weights from training subset to counter imbalance
    if args.class_weighted_loss:
        # collect labels from train loader once
        label_counts = None
        total_labels = 0
        for _, labels in train_loader:
            labels_np = labels.numpy()
            if label_counts is None:
                label_counts = np.bincount(labels_np, minlength=2)
            else:
                label_counts += np.bincount(labels_np, minlength=2)
            total_labels += labels_np.size
        # inverse frequency weights
        inv_freq = 1.0 / np.maximum(label_counts, 1)
        weights = torch.tensor(inv_freq / inv_freq.sum() * 2.0, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_valid_acc = 0.0
    best_path = os.path.join(args.out_dir, 'best_model.pth')
    last_path = os.path.join(args.out_dir, 'last_model.pth')

    history = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        valid_loss, valid_acc, v_true, v_prob, _ = evaluate(model, criterion, valid_loader, device)

        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Acc/train', train_acc, epoch)
            writer.add_scalar('Loss/valid', valid_loss, epoch)
            writer.add_scalar('Acc/valid', valid_acc, epoch)

        if conn:
            log_sqlite(conn, epoch, 'train', train_loss, train_acc)
            log_sqlite(conn, epoch, 'valid', valid_loss, valid_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'classes': classes}, last_path)
        if valid_acc >= best_valid_acc:
            best_valid_acc = valid_acc
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'classes': classes}, best_path)
            if not args.save_best_only:
                torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'classes': classes}, os.path.join(args.out_dir, f'epoch_{epoch}.pth'))

        print(f'Epoch {epoch}/{args.epochs} - Train loss {train_loss:.4f} acc {train_acc:.4f} | Valid loss {valid_loss:.4f} acc {valid_acc:.4f}')

    # Determine threshold
    used_threshold = None
    if args.auto_threshold and v_prob.size > 0 and v_prob.shape[1] >= 2:
        # grid search over unique prob values
        from sklearn.metrics import f1_score
        scores = []
        candidates = np.unique(v_prob[:, 1])
        for t in candidates:
            preds_t = (v_prob[:, 1] >= t).astype(int)
            scores.append((f1_score(v_true, preds_t, average='binary', pos_label=1), float(t)))
        used_threshold = max(scores, key=lambda x: x[0])[1] if scores else 0.5
    elif args.threshold is not None:
        used_threshold = args.threshold
    else:
        used_threshold = 0.5

    # Save chosen threshold
    with open(os.path.join(args.out_dir, 'threshold.txt'), 'w', encoding='utf-8') as f:
        f.write(str(used_threshold))

    test_loss, test_acc, y_true, y_prob, _ = evaluate(model, criterion, test_loader, device)
    print(f'Test - loss {test_loss:.4f} acc {test_acc:.4f} (threshold={used_threshold:.3f})')
    if writer:
        writer.add_scalar('Loss/test', test_loss, args.epochs)
        writer.add_scalar('Acc/test', test_acc, args.epochs)
        writer.close()
    if conn:
        log_sqlite(conn, args.epochs, 'test', test_loss, test_acc)
        conn.close()

    # Visualization outputs
    # 1) Curves
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label='Train Loss')
    plt.plot(history["valid_loss"], label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'loss_curve.png'))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_acc"], label='Train Acc')
    plt.plot(history["valid_acc"], label='Valid Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'acc_curve.png'))
    plt.close()

    # 2) Confusion matrix (binary classes expected: 0/1)
    if y_true.size > 0 and y_prob.size > 0:
        y_pred_thresh = (y_prob[:, 1] >= used_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_thresh)
        plt.figure(figsize=(4, 4))
        plt.imshow(cm, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xticks([0, 1], classes)
        plt.yticks([0, 1], classes)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha='center', va='center')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'confusion_matrix.png'))
        plt.close()

    # 3) ROC curve (binary; positive class index 1)
    if y_true.size > 0 and y_prob.size > 0 and y_prob.shape[1] >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'roc_curve.png'))
        plt.close()

    # 4) Grid of predictions from a batch
    try:
        images, labels = next(iter(test_loader))
        images = images.to(device)
        with torch.no_grad():
            probs = F.softmax(model(images), dim=1).cpu()
            preds = (probs[:, 1] >= used_threshold).to(torch.long)
        # denormalize for visualization
        grid = make_grid(images.cpu(), nrow=min(8, images.size(0)), normalize=True, scale_each=True)
        save_image(grid, os.path.join(args.out_dir, 'test_batch_grid.png'))
        # Also save a textual mapping
        with open(os.path.join(args.out_dir, 'test_batch_predictions.txt'), 'w', encoding='utf-8') as f:
            for i in range(images.size(0)):
                f.write(f'idx={i}\ttrue={classes[int(labels[i])]}\tpred={classes[int(preds[i])]}\tprob_face={probs[i,1].item():.3f}\n')
    except Exception:
        pass


if __name__ == '__main__':
    main()


