# Database Storage for Face Detection Project

## Overview

This project now includes comprehensive **SQLite database storage** for all training results, metrics, and metadata. This fulfills the requirement: *"visualisation et stockage des résultats et des méta-données dans une base de données"*.

## Database Schema

The database (`results.db`) contains the following tables:

### 1. **training_sessions**
Stores overall training session information:
- `session_id`: Unique identifier
- `timestamp`: When training started
- `epochs`, `batch_size`, `learning_rate`: Training hyperparameters
- `optimizer`, `architecture`: Model configuration
- `total_train_samples`, `total_valid_samples`, `total_test_samples`: Dataset sizes
- `best_valid_acc`, `final_test_acc`: Performance metrics
- `threshold`: Optimized decision threshold
- `config_json`: Full configuration as JSON

### 2. **epoch_metrics**
Stores metrics for each epoch and data split (train/valid/test):
- `session_id`: Links to training session
- `epoch`: Epoch number
- `split`: 'train', 'valid', or 'test'
- `loss`, `accuracy`, `precision`, `recall`, `f1_score`: Performance metrics

### 3. **model_checkpoints**
Metadata about saved model files:
- `session_id`: Links to training session
- `epoch`: When checkpoint was saved
- `checkpoint_path`: File location
- `file_size_bytes`: Model file size
- `is_best`: Whether this is the best model
- `valid_acc`: Validation accuracy at this checkpoint

### 4. **test_predictions**
Individual test sample predictions:
- `session_id`: Links to training session
- `sample_index`: Sample identifier
- `true_label`, `predicted_label`: Ground truth and prediction
- `confidence_class0`, `confidence_class1`: Model confidence scores
- `is_correct`: Whether prediction was correct

### 5. **confusion_matrices**
Confusion matrix results:
- `session_id`: Links to training session
- `split`: Data split ('train', 'valid', 'test')
- `threshold`: Decision threshold used
- `true_negative`, `false_positive`, `false_negative`, `true_positive`: Confusion matrix values

### 6. **visualizations**
Tracks all generated visualization files:
- `session_id`: Links to training session
- `viz_type`: Type of visualization (e.g., 'loss_curve', 'confusion_matrix', 'roc_curve')
- `file_path`: Location of the visualization file

## Usage

### Training with Database Storage

```bash
# Standard training (database enabled by default)
python train.py --train-dir train_images --test-dir test_images --epochs 15

# Training without database
python train.py --no-database --epochs 10

# Custom database path
python train.py --db-path ./my_results.db --epochs 10
```

### Testing with Database Storage

```bash
# Test and store results in database
python test.py --checkpoint ./artifacts/best_model.pth --save-predictions --save-errors

# Associate test with existing training session
python test.py --checkpoint ./artifacts/best_model.pth --session-id 1

# Test without database
python test.py --no-database --checkpoint ./artifacts/best_model.pth
```

### Querying the Database

Use the `query_db.py` utility to view stored results:

```bash
# List all training sessions
python query_db.py list

# Show detailed report for session ID 1
python query_db.py show 1

# Show latest training session
python query_db.py latest

# Show top 5 best models
python query_db.py best --top 5

# Export metrics to CSV
python query_db.py export --output my_metrics.csv

# Export specific session to CSV
python query_db.py export --session-id 1 --output session1_metrics.csv
```

## Example Queries

### Using Python API

```python
from database import ResultsDatabase

# Open database
with ResultsDatabase('./artifacts/results.db') as db:
    # Get all sessions
    sessions = db.get_all_sessions()
    
    # Get detailed session info
    session = db.get_session_summary(session_id=1)
    
    # Get training history
    history = db.get_training_history(session_id=1)
    
    # Get best models
    best_models = db.get_best_models(top_n=5)
    
    # Export to CSV
    db.export_to_csv('./metrics.csv', session_id=1)
```

### Direct SQL Queries

```bash
# Using SQLite command line
sqlite3 deep_learning_project/artifacts/results.db

# Example queries:
SELECT * FROM training_sessions ORDER BY best_valid_acc DESC LIMIT 5;
SELECT epoch, split, accuracy, f1_score FROM epoch_metrics WHERE session_id = 1;
SELECT * FROM confusion_matrices WHERE split = 'test';
```

## Benefits

1. **Persistent Storage**: All results stored permanently in structured format
2. **Easy Comparison**: Compare different training runs and hyperparameters
3. **Reproducibility**: Track exact configurations for each experiment
4. **Analysis**: Query and analyze results programmatically
5. **Export**: Easy export to CSV for further analysis or reporting
6. **Visualization Tracking**: Keep track of all generated plots and visualizations

## Database Location

Default: `deep_learning_project/artifacts/results.db`

The database is created automatically on first training run and can be queried at any time.
