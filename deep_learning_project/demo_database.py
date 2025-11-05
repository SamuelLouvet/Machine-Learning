#!/usr/bin/env python3
"""
Quick demo of database functionality
"""

from database import ResultsDatabase
import numpy as np

# Create and populate demo database
print("Creating demo database...")
db_path = './deep_learning_project/artifacts/demo_results.db'

with ResultsDatabase(db_path) as db:
    # Create a demo training session
    session_id = db.create_training_session({
        'epochs': 10,
        'batch_size': 64,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'architecture': 'CNN',
        'augment': True,
        'auto_threshold': True
    })
    
    print(f"Created demo session ID: {session_id}")
    
    # Add some demo epoch metrics
    for epoch in range(1, 11):
        # Simulated improving metrics
        train_acc = 0.7 + (epoch / 10) * 0.29
        valid_acc = 0.68 + (epoch / 10) * 0.30
        train_loss = 0.5 - (epoch / 10) * 0.45
        valid_loss = 0.52 - (epoch / 10) * 0.42
        
        db.add_epoch_metrics(session_id, epoch, 'train', {
            'loss': train_loss,
            'accuracy': train_acc,
            'precision': train_acc + 0.01,
            'recall': train_acc - 0.01,
            'f1_score': train_acc
        })
        
        db.add_epoch_metrics(session_id, epoch, 'valid', {
            'loss': valid_loss,
            'accuracy': valid_acc,
            'precision': valid_acc + 0.01,
            'recall': valid_acc - 0.02,
            'f1_score': valid_acc - 0.01
        })
    
    # Add test metrics
    db.add_epoch_metrics(session_id, 10, 'test', {
        'loss': 0.15,
        'accuracy': 0.97,
        'precision': 0.96,
        'recall': 0.85,
        'f1_score': 0.90
    })
    
    # Add confusion matrix
    confusion_matrix = [[6760, 71], [150, 647]]
    db.add_confusion_matrix(session_id, 'test', confusion_matrix, threshold=0.5)
    
    # Add some demo checkpoints
    db.add_model_checkpoint(session_id, 5, './artifacts/epoch_5.pth', is_best=False, valid_acc=0.91)
    db.add_model_checkpoint(session_id, 10, './artifacts/best_model.pth', is_best=True, valid_acc=0.98)
    
    # Update session with final results
    db.update_training_session(session_id, {
        'total_train_samples': 20000,
        'total_valid_samples': 5000,
        'total_test_samples': 7628,
        'best_valid_acc': 0.98,
        'final_test_acc': 0.97,
        'threshold': 0.8566
    })
    
    # Add some demo predictions (small sample)
    predictions = [
        {'sample_index': i, 'true_label': i % 2, 'predicted_label': i % 2, 
         'confidence_class0': 0.9 if i % 2 == 0 else 0.1,
         'confidence_class1': 0.1 if i % 2 == 0 else 0.9}
        for i in range(100)
    ]
    db.add_test_predictions(session_id, predictions)
    
    # Add visualizations
    db.add_visualization(session_id, 'loss_curve', './artifacts/loss_curve.png')
    db.add_visualization(session_id, 'acc_curve', './artifacts/acc_curve.png')
    db.add_visualization(session_id, 'confusion_matrix', './artifacts/confusion_matrix.png')
    db.add_visualization(session_id, 'roc_curve', './artifacts/roc_curve.png')
    
    print("\nDemo data added successfully!")
    
    # Query and display
    print("\n" + "="*70)
    print("DEMO SESSION SUMMARY")
    print("="*70)
    
    summary = db.get_session_summary(session_id)
    print(f"\nSession ID: {summary['session_id']}")
    print(f"Timestamp: {summary['timestamp']}")
    print(f"Configuration:")
    print(f"  Epochs: {summary['epochs']}")
    print(f"  Batch Size: {summary['batch_size']}")
    print(f"  Learning Rate: {summary['learning_rate']}")
    print(f"  Best Validation Accuracy: {summary['best_valid_acc']:.4f}")
    print(f"  Final Test Accuracy: {summary['final_test_acc']:.4f}")
    print(f"  Optimal Threshold: {summary['threshold']:.4f}")
    
    print(f"\nTotal Samples:")
    print(f"  Training: {summary['total_train_samples']}")
    print(f"  Validation: {summary['total_valid_samples']}")
    print(f"  Test: {summary['total_test_samples']}")
    
    print(f"\nStored {len(summary['metrics'])} epoch metrics")
    print(f"Stored {len(summary['checkpoints'])} model checkpoints")
    print(f"Stored {len(summary['confusion_matrices'])} confusion matrices")
    
    if summary['confusion_matrices']:
        print(f"\nTest Set Confusion Matrix (threshold={summary['threshold']:.4f}):")
        split, thresh, tn, fp, fn, tp = summary['confusion_matrices'][0]
        print(f"  True Negatives:  {tn:>6}")
        print(f"  False Positives: {fp:>6}")
        print(f"  False Negatives: {fn:>6}")
        print(f"  True Positives:  {tp:>6}")
    
    print("\n" + "="*70)
    print(f"\nDemo database created at: {db_path}")
    print("You can query it using:")
    print(f"  python deep_learning_project/query_db.py --db {db_path} list")
    print(f"  python deep_learning_project/query_db.py --db {db_path} show {session_id}")
    print("="*70 + "\n")

print("\n✅ Database functionality demo completed successfully!")
