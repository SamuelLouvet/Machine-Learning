# Deep Learning Face Detection (5IF OT2)

Projet de détection de visages utilisant un réseau de neurones convolutif (CNN) avec PyTorch.

## ✨ Fonctionnalités

- **CNN PyTorch** : Architecture convolutive à 2 couches
- **Data Augmentation** : Amélioration de la généralisation
- **Métriques Complètes** : Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Optimisation du Seuil** : Recherche automatique du meilleur seuil (F1-based)
- **Base de Données SQLite** : Stockage de tous les résultats et métadonnées
- **TensorBoard** : Visualisation en temps réel de l'entraînement
- **Analyse d'Erreurs** : Sauvegarde des échantillons mal classifiés

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 📁 Structure des Données

Placez vos images dans les dossiers avec sous-dossiers par classe (0 = non-visage, 1 = visage) :

```
deep_learning_project/
├── train_images/
│   ├── 0/  # Non-visages
│   └── 1/  # Visages
└── test_images/
    ├── 0/  # Non-visages
    └── 1/  # Visages
```

## 🎯 Utilisation

### 1. Entraînement

```bash
# Entraînement standard avec augmentation et optimisation du seuil
python deep_learning_project/train.py --epochs 15 --augment --auto-threshold

# Entraînement avec TensorBoard
python deep_learning_project/train.py --epochs 15 --augment --auto-threshold --tensorboard
```

**Options principales :**
- `--epochs` : Nombre d'époques (défaut: 10)
- `--batch-size` : Taille des batchs (défaut: 64)
- `--lr` : Learning rate (défaut: 0.001)
- `--augment` : Activer l'augmentation de données
- `--auto-threshold` : Optimisation automatique du seuil de décision
- `--tensorboard` : Activer TensorBoard
- `--use-imbalanced-sampler` : Gérer les classes déséquilibrées

### 2. Test et Évaluation

```bash
# Évaluation complète avec analyse d'erreurs et métriques complètes
python deep_learning_project/evaluate.py --save-errors --save-predictions
```

### 3. Consultation de la Base de Données

Tous les résultats sont stockés dans `deep_learning_project/artifacts/results.db`.

```bash
# Afficher la dernière session d'entraînement
python deep_learning_project/query_db.py latest

# Lister toutes les sessions
python deep_learning_project/query_db.py list

# Afficher les détails d'une session
python deep_learning_project/query_db.py show <session_id>

# Afficher les meilleurs modèles
python deep_learning_project/query_db.py best --top 5

# Exporter les métriques en CSV
python deep_learning_project/query_db.py export --output metrics.csv

# Visualiser la base de données
python deep_learning_project/view_database_sql.py
```

#### Structure de la Base de Données

La base de données SQLite contient 6 tables :

1. **training_sessions** : Informations globales (hyperparamètres, config, résultats finaux)
2. **epoch_metrics** : Métriques par époque et split (train/valid/test)
3. **model_checkpoints** : Métadonnées des modèles sauvegardés
4. **test_predictions** : Prédictions individuelles sur les échantillons de test
5. **confusion_matrices** : Résultats des matrices de confusion
6. **visualizations** : Fichiers de visualisation générés

#### Requêtes Python

```python
from database import ResultsDatabase

with ResultsDatabase('./deep_learning_project/artifacts/results.db') as db:
    # Récupérer toutes les sessions
    sessions = db.get_all_sessions()
    
    # Obtenir le résumé d'une session
    session = db.get_session_summary(session_id=1)
    
    # Historique d'entraînement
    history = db.get_training_history(session_id=1)
    
    # Meilleurs modèles
    best_models = db.get_best_models(top_n=5)
    
    # Export CSV
    db.export_to_csv('./metrics.csv', session_id=1)
```

## 📊 Architecture du Projet

```
deep_learning_project/
├── train.py              # Script d'entraînement
├── evaluate.py           # Script d'évaluation avec métriques complètes
├── net.py                # Architecture CNN
├── load_data.py          # Chargement et prétraitement des données
├── metrics.py            # Calcul des métriques
├── database.py           # Gestion de la base de données SQLite
├── query_db.py           # Outil CLI pour interroger la BD
├── view_database_sql.py  # Visualiseur SQL de la BD
└── artifacts/            # Sorties (générées, non versionnées)
    ├── results.db        # Base de données des résultats
    ├── best_model.pth    # Meilleur modèle
    ├── *.png             # Visualisations
    └── tb/               # Logs TensorBoard
```

## 📈 Résultats

Le projet stocke automatiquement dans la base de données :
- **Métriques par époque** : Loss, Accuracy, Precision, Recall, F1-Score (train + validation)
- **Checkpoints** : Sauvegarde des meilleurs modèles
- **Matrices de confusion** : Pour chaque évaluation
- **Prédictions** : Échantillons de test avec confiances
- **Visualisations** : Courbes, graphiques, analyse d'erreurs

## 💡 Notes

- **GPU automatique** : Utilise CUDA si disponible
- **Split automatique** : Données divisées en train/validation/test (80/20)
- **Métriques complètes** : F1-Score calculé pour train ET validation
- **Gestion déséquilibre** : Sampler optionnel pour classes déséquilibrées
- **Tous les artefacts** sont sauvegardés dans `artifacts/` (exclu du versioning via `.gitignore`)