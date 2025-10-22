Deep Learning Face Detection (5IF OT2)

Setup
- Install requirements: `pip install -r requirements.txt`

Data
- Place training images under `deep_learning_project/train_images/` and test images under `deep_learning_project/test_images/` in class subfolders (e.g., `0/` and `1/`).

Train
- From the `deep_learning_project` folder:
```
python train.py --epochs 10 --batch-size 64 --tensorboard --log-sqlite --use-imbalanced-sampler
```

Evaluate
```
python evaluate.py --checkpoint ./artifacts/best_model.pth
```

Notes
- Loaders are in `deep_learning_project/load_data.py` and support train/val/test splitting and an optional imbalanced sampler.
- Model is defined in `deep_learning_project/net.py`.

