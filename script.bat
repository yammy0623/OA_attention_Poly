@echo off

echo Running ORIGINAL...
python my_train.py --training_type original

echo Running GradCAM...
python my_train.py --training_type GradCAM

echo Running GradCAMPlusPlus...
python my_train.py --training_type GradCAMPlusPlus

echo Running ScoreCAM...
python my_train.py --training_type ScoreCAM

echo Running AblationCAM...
python my_train.py --training_type AblationCAM

echo Running LayerCAM...
python my_train.py --training_type LayerCAM

echo All CAM experiments done!
pause