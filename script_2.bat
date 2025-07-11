
echo Running ORIGINAL...
python my_train.py --training_type original

@REM echo Running GradCAMPlusPlus...
@REM python my_train_2.py --training_type GradCAMPlusPlus

@REM echo Running ScoreCAM...
@REM python my_train_2.py --training_type ScoreCAM

@REM echo Running AblationCAM...
@REM python my_train_2.py --training_type AblationCAM

@REM echo Running LayerCAM...
@REM python my_train_2.py --training_type LayerCAM

@REM echo Running GradCAM...
@REM python my_train_2.py --training_type GradCAM

echo All CAM experiments done!
pause