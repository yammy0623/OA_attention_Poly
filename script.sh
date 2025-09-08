# python train.py \
#   --debug \
#   --model_type MTLOrdinal_MultiTask \
#   --lossfcn_type CoralFocalLoss_MultiTask \
#   --predict_criteria Coral_Multitask \
#   --classweight_type inv \
#   --multitask_type all \
#   --feedback_type off \
#   --feedback_cam off \


python train.py \
  --model_type MTLOrdinal_MultiTask \
  --lossfcn_type CoralFocalLoss_MultiTask \
  --predict_criteria Coral_Multitask \
  --classweight_type inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \