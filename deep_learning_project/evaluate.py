import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from torchvision.utils import save_image

from net import Net
from load_data import create_data
from database import ResultsDatabase


def parse_args():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_checkpoint = os.path.join(script_dir, 'artifacts', 'best_model.pth')
    default_train_dir = os.path.join(script_dir, 'train_images')
    default_test_dir = os.path.join(script_dir, 'test_images')
    default_out_dir = os.path.join(script_dir, 'artifacts')
    default_db_path = os.path.join(script_dir, 'artifacts', 'results.db')
    
    parser = argparse.ArgumentParser(description='Test face detector CNN and generate comprehensive metrics')
    parser.add_argument('--checkpoint', default=default_checkpoint,
                        help='Path to model checkpoint')
    parser.add_argument('--train-dir', default=default_train_dir)
    parser.add_argument('--test-dir', default=default_test_dir)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--valid-size', type=float, default=0.2)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--image-size', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out-dir', default=default_out_dir,
                        help='Directory to save test results')
    parser.add_argument('--save-errors', action='store_true',
                        help='Save misclassified images')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save predictions to file')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Custom decision threshold (default: 0.5)')
    parser.add_argument('--db-path', default=default_db_path,
                        help='Database path for storing results')
    parser.add_argument('--session-id', type=int, default=None,
                        help='Existing training session ID to associate test results with')
    parser.add_argument('--no-database', action='store_true',
                        help='Disable database storage')
    return parser.parse_args()


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(cm, classes, save_path):
    """Plot and save confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label',
           title='Confusion Matrix')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrix saved to: {save_path}')


def save_error_samples(images, labels, predictions, probs, save_dir, max_samples=50):
    """Save misclassified samples for error analysis"""
    ensure_dir(save_dir)
    
    # Find misclassified indices
    errors = (predictions != labels)
    error_indices = np.where(errors)[0]
    
    if len(error_indices) == 0:
        print('No errors found!')
        return
    
    # Limit number of samples
    error_indices = error_indices[:max_samples]
    
    for idx in error_indices:
        img = images[idx]
        true_label = labels[idx]
        pred_label = predictions[idx]
        confidence = probs[idx, pred_label]
        
        filename = f'error_true{true_label}_pred{pred_label}_conf{confidence:.3f}_{idx}.png'
        save_image(img, os.path.join(save_dir, filename))
    
    print(f'Saved {len(error_indices)} error samples to: {save_dir}')


def evaluate_model(model, loader, device, threshold=0.5):
    """Evaluate model and return detailed metrics"""
    model.eval()
    
    all_images = []
    all_labels = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            # Apply custom threshold if provided
            if threshold != 0.5:
                # Use class 1 probability with custom threshold
                predicted = (probs[:, 1] >= threshold).long()
            else:
                _, predicted = torch.max(probs, 1)
            
            all_images.append(images.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            all_preds.append(predicted.cpu())
    
    # Concatenate all batches
    images_tensor = torch.cat(all_images)
    labels_np = torch.cat(all_labels).numpy()
    probs_np = torch.cat(all_probs).numpy()
    preds_np = torch.cat(all_preds).numpy()
    
    return images_tensor, labels_np, probs_np, preds_np


def compute_metrics(labels, predictions, probs):
    """Compute comprehensive classification metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(labels, predictions)
    metrics['precision'] = precision_score(labels, predictions, average='binary', pos_label=1, zero_division=0)
    metrics['recall'] = recall_score(labels, predictions, average='binary', pos_label=1, zero_division=0)
    metrics['f1'] = f1_score(labels, predictions, average='binary', pos_label=1, zero_division=0)
    
    # ROC-AUC (requires probabilities)
    try:
        metrics['roc_auc'] = roc_auc_score(labels, probs[:, 1])
    except Exception as e:
        print(f'Warning: Could not compute ROC-AUC: {e}')
        metrics['roc_auc'] = float('nan')
    
    # Per-class metrics
    metrics['precision_per_class'] = precision_score(labels, predictions, average=None, zero_division=0)
    metrics['recall_per_class'] = recall_score(labels, predictions, average=None, zero_division=0)
    metrics['f1_per_class'] = f1_score(labels, predictions, average=None, zero_division=0)
    
    return metrics


def print_results(metrics, cm, classes, total_samples):
    """Print formatted test results"""
    print('\n' + '='*60)
    print('TEST RESULTS')
    print('='*60)
    print(f'\nTotal test samples: {total_samples}')
    print(f'\nOverall Metrics:')
    print(f'  Accuracy:  {metrics["accuracy"]:.4f} ({metrics["accuracy"]*100:.2f}%)')
    print(f'  Precision: {metrics["precision"]:.4f}')
    print(f'  Recall:    {metrics["recall"]:.4f}')
    print(f'  F1-Score:  {metrics["f1"]:.4f}')
    print(f'  ROC-AUC:   {metrics["roc_auc"]:.4f}')
    
    print(f'\nPer-Class Metrics:')
    for i, cls in enumerate(classes):
        print(f'  Class {i} ({cls}):')
        print(f'    Precision: {metrics["precision_per_class"][i]:.4f}')
        print(f'    Recall:    {metrics["recall_per_class"][i]:.4f}')
        print(f'    F1-Score:  {metrics["f1_per_class"][i]:.4f}')
    
    print(f'\nConfusion Matrix:')
    print(f'                Predicted')
    print(f'                {classes[0]:>10} {classes[1]:>10}')
    for i, cls in enumerate(classes):
        print(f'  Actual {cls:>10}  {cm[i][0]:>10}  {cm[i][1]:>10}')
    
    # Calculate specific face detection metrics
    tn, fp, fn, tp = cm.ravel()
    print(f'\nDetailed Breakdown:')
    print(f'  True Negatives (Non-face correctly classified):  {tn}')
    print(f'  False Positives (Non-face misclassified as face): {fp}')
    print(f'  False Negatives (Face misclassified as non-face): {fn}')
    print(f'  True Positives (Face correctly classified):       {tp}')
    
    if (fp + tn) > 0:
        print(f'\n  False Positive Rate: {fp/(fp+tn):.4f}')
    if (fn + tp) > 0:
        print(f'  False Negative Rate: {fn/(fn+tp):.4f}')
    
    print('='*60 + '\n')


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    
    # Initialize database
    db = None
    session_id = args.session_id
    if not args.no_database:
        db = ResultsDatabase(args.db_path)
        if session_id is None:
            # Create a new test-only session
            session_id = db.create_training_session({
                'epochs': 0,
                'batch_size': args.batch_size,
                'learning_rate': 0.0,
                'optimizer': 'Test-Only',
                'architecture': 'CNN'
            })
            print(f"Database test session created: ID {session_id}")
    
    print(f'Loading model from: {args.checkpoint}')
    print(f'Using device: {args.device}')
    
    # Load data
    _, _, test_loader, classes = create_data(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        valid_size=args.valid_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    
    # Load model
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = Net().to(device)
    model.load_state_dict(checkpoint['model_state'])
    
    print(f'Model loaded successfully!')
    print(f'Classes: {classes}')
    
    # Get threshold
    threshold = args.threshold if args.threshold is not None else 0.5
    if threshold != 0.5:
        print(f'Using custom decision threshold: {threshold}')
    
    # Evaluate model
    print(f'\nEvaluating on test set...')
    images, labels, probs, predictions = evaluate_model(model, test_loader, device, threshold)
    
    # Compute metrics
    metrics = compute_metrics(labels, predictions, probs)
    cm = confusion_matrix(labels, predictions)
    
    # Print results
    print_results(metrics, cm, classes, len(labels))
    
    # Save confusion matrix plot
    cm_path = os.path.join(args.out_dir, 'test_confusion_matrix.png')
    plot_confusion_matrix(cm, classes, cm_path)
    
    # Store confusion matrix in database
    if db:
        db.add_confusion_matrix(session_id, 'test', cm.tolist(), threshold)
        db.add_visualization(session_id, 'test_confusion_matrix', cm_path)
    
    # Save classification report
    report_path = os.path.join(args.out_dir, 'test_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write('='*60 + '\n')
        f.write('FACE DETECTION TEST RESULTS\n')
        f.write('='*60 + '\n\n')
        f.write(f'Model: {args.checkpoint}\n')
        f.write(f'Test samples: {len(labels)}\n')
        f.write(f'Decision threshold: {threshold}\n\n')
        f.write(classification_report(labels, predictions, target_names=classes, digits=4))
        f.write('\n\nConfusion Matrix:\n')
        f.write(str(cm))
    print(f'Classification report saved to: {report_path}')
    
    # Save predictions if requested
    if args.save_predictions:
        pred_path = os.path.join(args.out_dir, 'test_predictions.txt')
        with open(pred_path, 'w') as f:
            f.write('Sample_ID,True_Label,Predicted_Label,Confidence_Class0,Confidence_Class1\n')
            for i in range(len(labels)):
                f.write(f'{i},{labels[i]},{predictions[i]},{probs[i, 0]:.6f},{probs[i, 1]:.6f}\n')
        print(f'Predictions saved to: {pred_path}')
        
        # Store predictions in database (limit to avoid too much data)
        if db:
            pred_list = [
                {
                    'sample_index': i,
                    'true_label': int(labels[i]),
                    'predicted_label': int(predictions[i]),
                    'confidence_class0': float(probs[i, 0]),
                    'confidence_class1': float(probs[i, 1])
                }
                for i in range(min(len(labels), 10000))  # Limit to 10k samples
            ]
            db.add_test_predictions(session_id, pred_list)
            print(f'Predictions stored in database (limited to {len(pred_list)} samples)')
    
    # Save error samples if requested
    if args.save_errors:
        error_dir = os.path.join(args.out_dir, 'test_errors')
        save_error_samples(images, labels, predictions, probs, error_dir)
        if db:
            db.add_visualization(session_id, 'test_errors', error_dir)
    
    # Update database with test results
    if db:
        db.update_training_session(session_id, {
            'total_test_samples': len(labels),
            'final_test_acc': metrics['accuracy'],
            'threshold': threshold
        })
        
        # Add test metrics
        db.add_epoch_metrics(session_id, 0, 'test', {
            'loss': 0.0,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1']
        })
        
        print(f'\n{"="*70}')
        print(f'Results saved to database: {args.db_path}')
        print(f'Session ID: {session_id}')
        print(f'{"="*70}\n')
        db.close()
    
    print('\nTesting complete!')


if __name__ == '__main__':
    main()

