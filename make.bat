@echo off
SET PY=python
SET DL=deep_learning_project

IF "%1"=="train" GOTO TRAIN
IF "%1"=="eval" GOTO EVAL
IF "%1"=="viz" GOTO VIZ
IF "%1"=="metrics" GOTO METRICS
IF "%1"=="clean" GOTO CLEAN
ECHO Usage: make.bat [train|eval|viz|metrics|clean]
GOTO END

:TRAIN
cd %DL% && %PY% train.py --epochs 10 --batch-size 64 --tensorboard --log-sqlite --use-imbalanced-sampler --class-weighted-loss --augment
GOTO END

:EVAL
cd %DL% && %PY% evaluate.py --checkpoint ./artifacts/best_model.pth
GOTO END

:VIZ
ECHO Visualizations saved under %DL%/artifacts: loss_curve.png, acc_curve.png, confusion_matrix.png, roc_curve.png, test_batch_grid.png
GOTO END

:METRICS
cd %DL% && %PY% metrics.py --db ./artifacts/metrics.sqlite --out ./artifacts/metrics.csv
GOTO END

:CLEAN
DEL /Q %DL%\artifacts\*.png 2> NUL
DEL /Q %DL%\artifacts\*.pth 2> NUL
DEL /Q %DL%\artifacts\*.sqlite 2> NUL
DEL /Q %DL%\artifacts\*.csv 2> NUL
GOTO END

:END

