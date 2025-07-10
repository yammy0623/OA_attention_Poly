echo Running GradCAM...

python my_inference_new.py --training_type GradCAM

python my_inference_new.py --training_type GradCAMPlusPlus

python my_inference_new.py --training_type LayerCAM

echo All CAM experiments done!
pause