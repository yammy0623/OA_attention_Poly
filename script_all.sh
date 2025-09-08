#!/bin/bash

# echo "Running ORIGINAL..."
# python my_train.py --training_type original

echo "Running GradCAM..."
python my_train_3.py --training_type GradCAM --pre_ckpt model_checkpoints_tnc_final 



python my_train_feedback_all.py --training_type LayerCAM --pre_ckpt model_checkpoints_tnc_final --feedback_type 1
python my_train_feedback_all.py --training_type LayerCAM --pre_ckpt model_checkpoints_tnc_final --feedback_type 2
python my_train_feedback_all.py --training_type LayerCAM --pre_ckpt model_checkpoints_tnc_final --feedback_type 3
python my_train_feedback_all.py --training_type LayerCAM --pre_ckpt model_checkpoints_tnc_final --feedback_type 4
python my_train_feedback_all.py --training_type LayerCAM --pre_ckpt model_checkpoints_tnc_final --feedback_type 5

# echo "Running GradCAMPlusPlus..."
# python my_train.py --training_type GradCAMPlusPlus --pre_ckpt model_checkpoints_tnc_final

# echo "Running ScoreCAM..."
# python my_train.py --training_type ScoreCAM

# echo "Running AblationCAM..."
# python my_train.py --training_type AblationCAM

# echo "Running LayerCAM..."
# python my_train.py --training_type LayerCAM

# echo "All CAM experiments done!"


# python my_train.py --training_type AblationCAM --pre_ckpt model_checkpoints_0710_epoch200_myckpt_128
# python my_train.py --training_type ScoreCAM --pre_ckpt model_checkpoints_0710_epoch200_myckpt_128

python my_train_feedback_all.py --training_type LayerCAM --pre_ckpt model_checkpoints_tnc_final --feedback_type 7

python my_train_feedback_all.py --training_type LayerCAM --pre_ckpt model_checkpoints_tnc_final --feedback_type 9 --note seq_predict

python my_train_feedback_all.py --training_type LayerCAM --pre_ckpt model_checkpoints_tnc_final --feedback_type 10 --note modify_kl

python my_train_feedback_all.py --training_type LayerCAM --pre_ckpt model_checkpoints_tnc_final --feedback_type 10 --note fb_9_modify_kl

python my_train_feedback_all.py --training_type LayerCAM --pre_ckpt model_checkpoints_tnc_final --feedback_type 11 --note org_model_ordinal_loss


python my_train_feedback_all.py --training_type LayerCAM --pre_ckpt model_checkpoints_tnc_final --feedback_type 11 --note org_model_ordinal_loss_scaleup_cls1

python my_train_feedback_all.py --training_type LayerCAM --pre_ckpt model_checkpoints_tnc_final --feedback_type 12 --note org_model_focal_loss

python my_train_feedback_all.py --training_type LayerCAM --pre_ckpt model_checkpoints_tnc_final --feedback_type 12 --note org_model_focal_loss_scaleup_cls1


python my_train_feedback_all.py --training_type LayerCAM --pre_ckpt model_checkpoints_tnc_final --feedback_type 11 --note org_model_ordinal_loss_ENS_classweight

python my_train_feedback_all.py --training_type LayerCAM --pre_ckpt model_checkpoints_tnc_final --feedback_type 13 --note org_model_ordinal_loss_ENS_classweight

python my_train_feedback_all.py --training_type LayerCAM --pre_ckpt model_checkpoints_tnc_final --feedback_type 13 --note org_model_ordinal_loss_ENS_classweight_scaleup_cls1

python my_train_feedback_all.py --training_type original --pre_ckpt model_checkpoints_tnc_final --feedback_type 14 --note org_model_ordinal_loss_multitask_kl_jsml

python my_train_feedback_all.py --training_type original --pre_ckpt model_checkpoints_tnc_final --feedback_type 14 --note org_model_ordinal_loss_multitask_kl_all_oarsi


# ========================
python train.py \
  --model_type OrdinalMTL \
  --lossfcn_type CoralLossWeighted \
  --multitask_type all \
  --predict_criteria Coral_Multitask \
  --feedback_type off \
  --feedback_cam off \
  --seed 0 \
#   --pre_ckpt ./checkpoints/pretrained.ckpt \
#   --note ""
