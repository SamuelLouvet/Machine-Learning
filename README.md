Deep Learning Face Detection (5IF OT2)

## Features
- **CNN-based face detection** using PyTorch
- **Data augmentation** for improved generalization
- **Imbalanced dataset handling** with custom samplers
- **Automatic threshold optimization** for best F1-score
- **Comprehensive metrics** (accuracy, precision, recall, F1, ROC-AUC)
- **TensorBoard logging** for real-time monitoring
- **SQLite database storage** for all results and metadata
- **Visualization tools** (loss/accuracy curves, confusion matrices, ROC curves)

## Setup
- Install requirements: `pip install -r requirements.txt`

## Data
- Place training images under `deep_learning_project/train_images/` and test images under `deep_learning_project/test_images/` in class subfolders (e.g., `0/` and `1/`).

## Train
- From the project root:
```bash
# Basic training with database storage
python deep_learning_project/train.py --epochs 15 --batch-size 64 --tensorboard --augment --auto-threshold

# Advanced training with all features
python deep_learning_project/train.py \
  --train-dir deep_learning_project/train_images \
  --test-dir deep_learning_project/test_images \
  --epochs 15 \
  --batch-size 64 \
  --lr 0.001 \
  --tensorboard \
  --augment \
  --auto-threshold \
  --use-imbalanced-sampler
```

## Test
```bash
# Comprehensive testing with error analysis
python deep_learning_project/test.py \
  --checkpoint deep_learning_project/artifacts/best_model.pth \
  --save-errors \
  --save-predictions

# Test with custom threshold
python deep_learning_project/test.py \
  --checkpoint deep_learning_project/artifacts/best_model.pth \
  --threshold 0.8566
```

## Evaluate
```bash
# Quick evaluation on validation/test sets
python deep_learning_project/evaluate.py \
  --checkpoint deep_learning_project/artifacts/best_model.pth
```

## Database Features
All training and test results are automatically stored in SQLite database (`artifacts/results.db`).

### Query Database
```bash
# List all training sessions
python deep_learning_project/query_db.py list

# Show detailed session report
python deep_learning_project/query_db.py show <session_id>

# Show latest training session
python deep_learning_project/query_db.py latest

# Show top 5 best models
python deep_learning_project/query_db.py best --top 5

# Export metrics to CSV
python deep_learning_project/query_db.py export --output metrics.csv
```

See `deep_learning_project/DATABASE_README.md` for detailed database documentation.

## Project Structure
```
deep_learning_project/
├── train.py           # Training script with database integration
├── test.py            # Testing script with comprehensive metrics
├── evaluate.py        # Quick evaluation script
├── net.py             # CNN architecture definition
├── load_data.py       # Data loading and preprocessing
├── database.py        # Database management module
├── query_db.py        # Database query utility
├── metrics.py         # Metrics computation and export
├── demo_database.py   # Database functionality demo
├── DATABASE_README.md # Database documentation
└── artifacts/         # Output directory
    ├── results.db     # SQLite database with all results
    ├── best_model.pth # Best model checkpoint
    ├── *.png          # Visualizations
    └── *.txt          # Reports and predictions
```

## Notes
- Model architecture is defined in `deep_learning_project/net.py`
- Data loaders support train/val/test splitting with optional imbalanced sampler
- All results are stored in the database for easy comparison and analysis
- TensorBoard logs are saved to `artifacts/tb/` when `--tensorboard` is enabled

