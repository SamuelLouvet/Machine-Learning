import argparse
import os
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
from database import ResultsDatabase


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
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--db-path', default='./artifacts/results.db', help='Database path for storing results')
    parser.add_argument('--no-database', action='store_true', help='Disable database storage')
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


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    device = torch.device(args.device)

    # Display device information
    print(f"=" * 60)
    print(f"Training Configuration")
    print(f"=" * 60)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"=" * 60)
    print()

    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, 'tb')) if args.tensorboard else None
    
    # Initialize database
    db = None
    session_id = None
    if not args.no_database:
        db = ResultsDatabase(args.db_path)
        session_id = db.create_training_session({
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'optimizer': 'Adam',
            'architecture': 'CNN',
            'weight_decay': args.weight_decay,
            'use_imbalanced_sampler': args.use_imbalanced_sampler,
            'class_weighted_loss': args.class_weighted_loss,
            'augment': args.augment,
            'auto_threshold': args.auto_threshold,
            'seed': args.seed
        })
        print(f"Database session created: ID {session_id}")

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
        all_train_labels = []
        all_train_preds = []
        
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
            
            # Collect labels and predictions for F1 calculation
            all_train_labels.append(labels.cpu().numpy())
            all_train_preds.append(predicted.cpu().numpy())

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        
        # Concatenate all training predictions and labels
        train_labels_np = np.concatenate(all_train_labels)
        train_preds_np = np.concatenate(all_train_preds)

        valid_loss, valid_acc, v_true, v_prob, v_preds = evaluate(model, criterion, valid_loader, device)

        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Acc/train', train_acc, epoch)
            writer.add_scalar('Loss/valid', valid_loss, epoch)
            writer.add_scalar('Acc/valid', valid_acc, epoch)

        # Calculate F1-score and other metrics for both train and validation
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        train_precision = precision_score(train_labels_np, train_preds_np, average='binary', pos_label=1, zero_division=0)
        train_recall = recall_score(train_labels_np, train_preds_np, average='binary', pos_label=1, zero_division=0)
        train_f1 = f1_score(train_labels_np, train_preds_np, average='binary', pos_label=1, zero_division=0)
        
        valid_precision = precision_score(v_true, v_preds, average='binary', pos_label=1, zero_division=0) if len(v_true) > 0 else 0.0
        valid_recall = recall_score(v_true, v_preds, average='binary', pos_label=1, zero_division=0) if len(v_true) > 0 else 0.0
        valid_f1 = f1_score(v_true, v_preds, average='binary', pos_label=1, zero_division=0) if len(v_true) > 0 else 0.0
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)
        
        # Store metrics in database
        if db:
            train_metrics = {
                'loss': train_loss,
                'accuracy': train_acc,
                'precision': train_precision,
                'recall': train_recall,
                'f1_score': train_f1
            }
            db.add_epoch_metrics(session_id, epoch, 'train', train_metrics)
            
            valid_metrics = {
                'loss': valid_loss,
                'accuracy': valid_acc,
                'precision': valid_precision,
                'recall': valid_recall,
                'f1_score': valid_f1
            }
            db.add_epoch_metrics(session_id, epoch, 'valid', valid_metrics)

        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'classes': classes}, last_path)
        if valid_acc >= best_valid_acc:
            best_valid_acc = valid_acc
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'classes': classes}, best_path)
            if db:
                db.add_model_checkpoint(session_id, epoch, best_path, is_best=True, valid_acc=valid_acc)
            if not args.save_best_only:
                epoch_path = os.path.join(args.out_dir, f'epoch_{epoch}.pth')
                torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'classes': classes}, epoch_path)
                if db:
                    db.add_model_checkpoint(session_id, epoch, epoch_path, is_best=False, valid_acc=valid_acc)

        print(f'Epoch {epoch}/{args.epochs} - '
              f'Train: loss {train_loss:.4f} acc {train_acc:.4f} f1 {train_f1:.4f} | '
              f'Valid: loss {valid_loss:.4f} acc {valid_acc:.4f} f1 {valid_f1:.4f}')

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

    test_loss, test_acc, y_true, y_prob, y_preds = evaluate(model, criterion, test_loader, device)
    print(f'Test - loss {test_loss:.4f} acc {test_acc:.4f} (threshold={used_threshold:.3f})')
    
    # Store test metrics and update session
    if db:
        from sklearn.metrics import precision_score, recall_score, f1_score
        test_metrics = {
            'loss': test_loss,
            'accuracy': test_acc,
            'precision': precision_score(y_true, y_preds, average='binary', pos_label=1, zero_division=0) if len(y_true) > 0 else 0.0,
            'recall': recall_score(y_true, y_preds, average='binary', pos_label=1, zero_division=0) if len(y_true) > 0 else 0.0,
            'f1_score': f1_score(y_true, y_preds, average='binary', pos_label=1, zero_division=0) if len(y_true) > 0 else 0.0
        }
        db.add_epoch_metrics(session_id, args.epochs, 'test', test_metrics)
        
        # Update session with final results
        db.update_training_session(session_id, {
            'total_train_samples': len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 0,
            'total_valid_samples': len(valid_loader.dataset) if hasattr(valid_loader, 'dataset') else 0,
            'total_test_samples': len(test_loader.dataset) if hasattr(test_loader, 'dataset') else 0,
            'best_valid_acc': best_valid_acc,
            'final_test_acc': test_acc,
            'threshold': used_threshold
        })
    
    if writer:
        writer.add_scalar('Loss/test', test_loss, args.epochs)
        writer.add_scalar('Acc/test', test_acc, args.epochs)
        writer.close()


    # Visualization outputs
    # 1) Curves
    loss_curve_path = os.path.join(args.out_dir, 'loss_curve.png')
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label='Train Loss')
    plt.plot(history["valid_loss"], label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_curve_path)
    plt.close()
    if db:
        db.add_visualization(session_id, 'loss_curve', loss_curve_path)

    acc_curve_path = os.path.join(args.out_dir, 'acc_curve.png')
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_acc"], label='Train Acc')
    plt.plot(history["valid_acc"], label='Valid Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_curve_path)
    plt.close()
    if db:
        db.add_visualization(session_id, 'acc_curve', acc_curve_path)

    # 2) Confusion matrix (binary classes expected: 0/1)
    if y_true.size > 0 and y_prob.size > 0:
        y_pred_thresh = (y_prob[:, 1] >= used_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_thresh)
        cm_path = os.path.join(args.out_dir, 'confusion_matrix.png')
        plt.figure(figsize=(4, 4))
        plt.imshow(cm, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xticks([0, 1], classes)
        plt.yticks([0, 1], classes)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha='center', va='center')
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        
        # Store confusion matrix in database
        if db:
            db.add_confusion_matrix(session_id, 'test', cm.tolist(), used_threshold)
            db.add_visualization(session_id, 'confusion_matrix', cm_path)

    # 3) ROC curve (binary; positive class index 1)
    if y_true.size > 0 and y_prob.size > 0 and y_prob.shape[1] >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        roc_path = os.path.join(args.out_dir, 'roc_curve.png')
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()
        if db:
            db.add_visualization(session_id, 'roc_curve', roc_path)

    # 4) Grid of predictions from a batch
    try:
        images, labels = next(iter(test_loader))
        images = images.to(device)
        with torch.no_grad():
            probs = F.softmax(model(images), dim=1).cpu()
            preds = (probs[:, 1] >= used_threshold).to(torch.long)
        # denormalize for visualization
        grid = make_grid(images.cpu(), nrow=min(8, images.size(0)), normalize=True, scale_each=True)
        grid_path = os.path.join(args.out_dir, 'test_batch_grid.png')
        save_image(grid, grid_path)
        if db:
            db.add_visualization(session_id, 'test_batch_grid', grid_path)
        # Also save a textual mapping
        with open(os.path.join(args.out_dir, 'test_batch_predictions.txt'), 'w', encoding='utf-8') as f:
            for i in range(images.size(0)):
                f.write(f'idx={i}\ttrue={classes[int(labels[i])]}\tpred={classes[int(preds[i])]}\tprob_face={probs[i,1].item():.3f}\n')
    except Exception:
        pass
    
    # Close database and print summary
    if db:
        print(f'\n{"="*70}')
        print(f'Results saved to database: {args.db_path}')
        print(f'Session ID: {session_id}')
        print(f'{"="*70}\n')
        db.close()


if __name__ == '__main__':
    main()
