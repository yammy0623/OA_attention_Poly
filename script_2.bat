@REM echo Running GradCAM...
@REM python my_train.py --training_type GradCAM

@REM echo Running GradCAMPlusPlus...
@REM python my_train.py --training_type GradCAMPlusPlus

@REM echo Running ScoreCAM...
@REM python my_train.py --training_type ScoreCAM

@REM echo Running AblationCAM...
@REM python my_train.py --training_type AblationCAM

echo Running LayerCAM...
python my_train.py --training_type LayerCAM

echo All CAM experiments done!
pause