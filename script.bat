@echo off

@REM echo Running ORIGINAL...
@REM python my_train.py --training_type original

echo Running GradCAM...
python my_train.py --training_type GradCAM

echo Running GradCAMPlusPlus...
python my_train.py --training_type GradCAMPlusPlus

@REM echo Running ScoreCAM...
@REM python my_train.py --training_type ScoreCAM

@REM echo Running AblationCAM...
@REM python my_train.py --training_type AblationCAM

@REM echo Running LayerCAM...
@REM python my_train.py --training_type LayerCAM

echo All CAM experiments done!
pause