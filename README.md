# Deep Learning Face Detection (5IF OT2)
Projet de détection de visages utilisant un réseau de neurones convolutif (CNN) avec PyTorch.
## 🚀 Installation
```bash
pip install -r requirements.txt
```
## 📁 Structure des données
Placez vos images dans les dossiers suivants avec des sous-dossiers par classe (0 et 1) :
- `deep_learning_project/train_images/` - Images d'entraînement
- `deep_learning_project/test_images/` - Images de test
Exemple :
```
train_images/
  ├── 0/  (non-visages)
  └── 1/  (visages)
test_images/
  ├── 0/  (non-visages)
  └── 1/  (visages)
```
## 🎯 Commandes principales
### 1. Entraînement (Train)
Entraîner le modèle sur GPU avec les paramètres par défaut :
```bash
cd deep_learning_project
python train.py --epochs 10 --batch-size 64 --num-workers 0
```
**Options principales :**
- `--epochs` : Nombre d'époques (défaut: 10)
- `--batch-size` : Taille des batchs (défaut: 64)
- `--lr` : Learning rate (défaut: 0.001)
- `--num-workers` : Nombre de workers pour le DataLoader (défaut: 0 pour Windows)
- `--device` : Device à utiliser (défaut: cuda si disponible, sinon cpu)
- `--augment` : Activer l'augmentation de données
- `--tensorboard` : Activer TensorBoard pour visualisation
- `--use-imbalanced-sampler` : Utiliser un sampler pour données déséquilibrées
- `--class-weighted-loss` : Utiliser des poids de classe pour la loss
**Sorties :**
- Modèles sauvegardés dans `artifacts/` (best_model.pth, last_model.pth, epoch_*.pth)
- Courbes de loss et accuracy (loss_curve.png, acc_curve.png)
- Matrice de confusion (confusion_matrix.png)
- Courbe ROC (roc_curve.png)
- Grille de prédictions (test_batch_grid.png)
### 2. Évaluation (Eval)
Évaluer un modèle sauvegardé sur les ensembles de validation et test :
```bash
cd deep_learning_project
python evaluate.py --checkpoint ./artifacts/best_model.pth
```
**Options :**
- `--checkpoint` : Chemin vers le fichier .pth du modèle (requis)
- `--batch-size` : Taille des batchs (défaut: 64)
- `--device` : Device à utiliser (défaut: cuda si disponible)
**Affiche :**
- Loss et accuracy sur validation
- Loss et accuracy sur test
### 3. Métriques avancées (Metrics)
Calculer les métriques F1-Score et AUC-ROC sur validation et test :
```bash
cd deep_learning_project
python metrics.py --checkpoint ./artifacts/best_model.pth
```
**Options :**
- `--checkpoint` : Chemin vers le modèle (défaut: ./artifacts/best_model.pth)
- `--batch-size` : Taille des batchs (défaut: 64)
- `--num-workers` : Nombre de workers (défaut: 0)
- `--device` : Device à utiliser (défaut: cuda si disponible)
**Affiche :**
- F1-Score (moyenne harmonique précision/rappel)
- AUC-ROC (aire sous la courbe ROC)
## 📊 Exemple de workflow complet
```bash
# 1. Se placer dans le dossier du projet
cd deep_learning_project
# 2. Entraîner le modèle
python train.py --epochs 10 --num-workers 0
# 3. Évaluer les performances
python evaluate.py --checkpoint ./artifacts/best_model.pth
# 4. Calculer les métriques avancées
python metrics.py
```
## 🔧 Fichiers principaux
- `train.py` - Script d'entraînement du modèle
- `evaluate.py` - Script d'évaluation sur validation/test
- `metrics.py` - Calcul de métriques avancées (F1, AUC)
- `net.py` - Architecture du réseau CNN
- `load_data.py` - Chargement des données avec train/val/test split
## 💡 Notes
- Le projet utilise automatiquement le GPU (CUDA) s'il est disponible
- Les données sont automatiquement divisées en train/validation/test
- Support optionnel d'un sampler pour gérer les classes déséquilibrées
- Tous les artefacts (modèles, graphiques) sont sauvegardés dans `artifacts/`
