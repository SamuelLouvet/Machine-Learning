import argparse
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from net import Net
from load_data import create_data


def parse_args():
    p = argparse.ArgumentParser(description='Compute F1/AUC metrics from checkpoint')
    p.add_argument('--checkpoint', default='./artifacts/best_model.pth')
    p.add_argument('--train-dir', default='./train_images')
    p.add_argument('--test-dir', default='./test_images')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--valid-size', type=float, default=0.2)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--image-size', type=int, default=32)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def compute_split_metrics(split: str, loader, model, device) -> Optional[Dict[str, float]]:
    """
    Calcule les métriques F1-score et AUC pour un ensemble de données.

    Args:
        split: Nom de l'ensemble ('train', 'valid', 'test')
        loader: DataLoader PyTorch
        model: Modèle entraîné
        device: Device (cuda/cpu)

    Returns:
        Dictionnaire contenant f1 et auc
    """
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

    device = torch.device(args.device)

    print("=" * 60)
    print("Chargement du modèle et calcul des métriques...")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print()

    # Charger le modèle
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = Net().to(device)
    model.load_state_dict(checkpoint['model_state'])

    # Charger les données
    train_loader, valid_loader, test_loader, _ = create_data(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        valid_size=args.valid_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    # Calculer les métriques
    print("Calcul des métriques...")
    valid_metrics = compute_split_metrics('valid', valid_loader, model, device)
    test_metrics = compute_split_metrics('test', test_loader, model, device)

    print("\n" + "=" * 60)
    print("RÉSULTATS DES MÉTRIQUES")
    print("=" * 60)

    print("\n📊 Validation Set:")
    if valid_metrics:
        print(f"  F1-Score: {valid_metrics['f1']:.4f}")
        print(f"  AUC-ROC:  {valid_metrics['auc']:.4f}")
    else:
        print("  Aucune métrique disponible")

    print("\n📊 Test Set:")
    if test_metrics:
        print(f"  F1-Score: {test_metrics['f1']:.4f}")
        print(f"  AUC-ROC:  {test_metrics['auc']:.4f}")
    else:
        print("  Aucune métrique disponible")

    print("\n" + "=" * 60)
    print("\nExplication des métriques:")
    print("  • F1-Score: Moyenne harmonique de la précision et du rappel (0-1)")
    print("  • AUC-ROC:  Aire sous la courbe ROC - capacité de discrimination (0-1)")
    print("=" * 60)


if __name__ == '__main__':
    main()
