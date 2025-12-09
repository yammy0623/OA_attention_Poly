python train.py \
  --model_type MILOrdinal_MultiTask \
  --lossfcn_type CoralFocalLoss_MultiTask \
  --predict_criteria Coral_Multitask \
  --classweight_type inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train1

python inference.py \
  --current_ckpt model_checkpoints_20250911_011624_epoch200_MILOrdinal_MultiTask_LCFLoM_MKJ_C0_Fo_lr1e-04_b16 \
  --model_type MILOrdinal_MultiTask \
  --lossfcn_type CoralFocalLoss_MultiTask \
  --predict_criteria Coral_Multitask \
  --classweight_type inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train1



 python train.py \
  --model_type MILOrdinal_MultiTask \
  --lossfcn_type CoralFocalLoss_MultiTask \
  --predict_criteria Coral_Multitask \
  --classweight_type effective \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train2



 python train.py \
  --model_type MILOrdinal_MultiTask \
  --lossfcn_type CoralFocalLoss_MultiTask \
  --predict_criteria Coral_Multitask \
  --classweight_type balanace_sampling \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train3

  
python inference.py \
  --current_ckpt model_checkpoints_20250911_032024_epoch200_MILOrdinal_MultiTask_LCFLoM_MKJ_C0_Fo_lr1e-04_b16 \
  --model_type MILOrdinal_MultiTask \
  --lossfcn_type CoralFocalLoss_MultiTask \
  --predict_criteria Coral_Multitask \
  --classweight_type balanace_sampling \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train3


# Baseline with MultiTask
python train.py \
  --model_type MIL_MultiTask \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train7

python inference.py \
  --current_ckpt model_checkpoints_20250911_110619_epoch200_MIL_MultiTask_LCEoM_MKJ_C0_Fo_lr1e-04_b16 \
  --model_type MIL_MultiTask \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train7


python train.py \
  --model_type MIL_MultiTask \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train8

python inference.py \
  --current_ckpt model_checkpoints_20250911_110653_epoch200_MIL_MultiTask_LCEoM_MA_C0_Fo_lr1e-04_b16 \
  --model_type MIL_MultiTask \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train8

# Use GAP and GMP for different tasks
python train.py \
  --model_type MIL_wGP_MultiTask \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train9

python inference.py \
  --current_ckpt model_checkpoints_20250915_015710_epoch200_MIL_wGP_MultiTask_LCEoM_MKJ_C0_Fo_lr1e-04_b16 \
  --model_type MIL_wGP_MultiTask \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train9

python train.py \
  --model_type MIL_wGP_MultiTask \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train10

python inference.py \
  --current_ckpt model_checkpoints_20250915_015731_epoch200_MIL_wGP_MultiTask_LCEoM_MA_C0_Fo_lr1e-04_b16 \
  --model_type MIL_wGP_MultiTask \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train10


# train metrics with different weighted
python train.py \
  --model_type MIL_wGP_MultiTask \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train11


python inference.py \
  --current_ckpt model_checkpoints_20250924_093219_epoch200_MIL_wGP_MultiTask_LCFLMB_MA_C0_Fo_lr1e-04_b16 \
  --model_type MIL_wGP_MultiTask \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train11



python train.py \
  --model_type MIL_wGP_MultiTask \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train12

python inference.py \
  --current_ckpt model_checkpoints_20250924_093523_epoch200_MIL_wGP_MultiTask_LCFLMB_MKJ_C0_Fo_lr1e-04_b16 \
  --model_type MIL_wGP_MultiTask \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train12


python train.py \
  --model_type MIL_MultiTask \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train13

python inference.py \
  --current_ckpt model_checkpoints_20250924_093621_epoch200_MIL_MultiTask_LCFLMB_MA_C0_Fo_lr1e-04_b16 \
  --model_type MIL_MultiTask \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train13


# don't mask out the -999, set them to 0
python train.py \
  --model_type MIL_wGP_MultiTask \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train11_2

python inference.py \
  --current_ckpt model_checkpoints_20250924_142416_epoch200_MIL_wGP_MultiTask_LCFLMB_MA_C0_Fo_lr1e-04_b16 \
  --model_type MIL_wGP_MultiTask \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train11_2


python train.py \
  --model_type MIL_wGP_MultiTask \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train12_2

python inference.py \
  --current_ckpt model_checkpoints_20250924_142433_epoch200_MIL_wGP_MultiTask_LCFLMB_MKJ_C0_Fo_lr1e-04_b16 \
  --model_type MIL_wGP_MultiTask \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train12_2

python train.py \
  --debug \
  --model_type MIL_MultiTask \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train13_2

python inference.py \
  --current_ckpt model_checkpoints_20250924_142537_epoch200_MIL_MultiTask_LCFLMB_MA_C0_Fo_lr1e-04_b16 \
  --model_type MIL_MultiTask \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train13_2


# use imedslab classification head => it works!!!
python train.py \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train14

python inference.py \
  --current_ckpt model_checkpoints_20250924_210420_epoch200_MIL_MultiTask_imedslab_LCEoM_MKJ_C0_Fo_lr1e-04_b16 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train14


python train.py \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train15


python inference.py \
  --current_ckpt model_checkpoints_20250924_205632_epoch200_MIL_MultiTask_imedslab_LCEoM_MA_C0_Fo_lr1e-04_b16 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train15

# change it to 128*128
python train.py \
  --debug \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train16_1d

python train.py \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train16_2

# Condiser class weights normalization for CE with 100*100 (current final results)
python train.py \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train17_classweight_norm_CE_kl_jsn

python inference.py \
  --current_ckpt model_checkpoints_20250925_111435_epoch200_MIL_MultiTask_imedslab_LCEoM_MKJ_C0_Fo_lr1e-04_b16\
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train17_classweight_norm_CE_kl_jsn

python train.py \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv  \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train17_classweight_norm_CE_all

python inference.py \
  --current_ckpt model_checkpoints_20250925_111508_epoch200_MIL_MultiTask_imedslab_LCEoM_MA_C0_Fo_lr1e-04_b16_GOOD\
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv  \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train17_classweight_norm_CE_all

# 
python train.py \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train17_classweight_norm_CE_kl_jsn

# change to ordinal regression
python inference.py \
  --current_ckpt model_checkpoints_20250926_150707_epoch200_MILOrdinal_MultiTask_imedslab_LBCEWL_MKJ_C0_Fo_lr1e-04_b16 \
  --model_type MILOrdinal_MultiTask_imedslab \
  --lossfcn_type BCEWithLogitsLoss_MultiTask \
  --predict_criteria ordinal \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train18_classweight_norm_ordinal_CE_kl_jsn

python train.py \
  --model_type MILOrdinal_MultiTask_imedslab \
  --lossfcn_type BCEWithLogitsLoss_MultiTask \
  --predict_criteria ordinal \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train18_classweight_norm_ordinal_CE_kl_jsn

# Use CoralFocalLoss with imedslab => Fail
python train.py \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Coral_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train19_classweight_norm_CE_kl_jsn

python inference.py \
  --current_ckpt model_checkpoints_20250929_101136_epoch200_MIL_MultiTask_imedslab_LCFLMB_MKJ_C0_Fo_lr1e-04_b16 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train19_classweight_norm_CE_kl_jsn

python train.py \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Coral_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train19_classweight_norm_CE_all

python inference.py \
  --current_ckpt model_checkpoints_20250929_101209_epoch200_MIL_MultiTask_imedslab_LCFLMB_MA_C0_Fo_lr1e-04_b16 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train19_classweight_norm_CE_all

# Retrain
python train.py \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Coral_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train19_classweight_norm_CE_kl_jsn

python inference.py \
  --current_ckpt model_checkpoints_20250929_141722_epoch200_MIL_MultiTask_imedslab_LCFLMB_MKJ_C0_Fo_lr1e-04_b16 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train19_classweight_norm_CE_kl_jsn


python train.py \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Coral_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train19_classweight_norm_CE_all

python inference.py \
  --current_ckpt model_checkpoints_20250929_141742_epoch200_MIL_MultiTask_imedslab_LCFLMB_MA_C0_Fo_lr1e-04_b16 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Coral_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train19_classweight_norm_CE_all


# Retrain with CoralLoss
python train.py \
  --model_type MILOrdinal_MultiTask_imedslab \
  --lossfcn_type CoralLoss_MultiTask \
  --predict_criteria Coral_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train20_classweight_norm_CE_kl_jsn

python inference.py \
  --current_ckpt model_checkpoints_20250929_161151_epoch200_MILOrdinal_MultiTask_imedslab_LCLoM_MKJ_C0_Fo_lr1e-04_b16 \
  --model_type MILOrdinal_MultiTask_imedslab \
  --lossfcn_type CoralLoss_MultiTask \
  --predict_criteria Coral_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train20_classweight_norm_CE_kl_jsn

python train.py \
  --model_type MILOrdinal_MultiTask_imedslab \
  --lossfcn_type CoralLoss_MultiTask \
  --predict_criteria Coral_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train20_classweight_norm_CE_all

python inference.py \
  --current_ckpt model_checkpoints_20250929_161214_epoch200_MILOrdinal_MultiTask_imedslab_LCLoM_MA_C0_Fo_lr1e-04_b16 \
  --model_type MILOrdinal_MultiTask_imedslab \
  --lossfcn_type CoralLoss_MultiTask \
  --predict_criteria Coral_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train20_classweight_norm_CE_all

# Retrain with 128*128
python train.py \
  --debug \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Coral_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train19_classweight_norm_CE_kl_jsn_128

python inference.py \
  --current_ckpt model_checkpoints_20250929_220414_epoch200_MIL_MultiTask_imedslab_LCFLMB_MKJ_C0_Fo_lr1e-04_b16 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Coral_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train19_classweight_norm_CE_kl_jsn_128

python train.py \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Coral_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train19_classweight_norm_CE_all_128

python inference.py \
  --current_ckpt model_checkpoints_20250929_220508_epoch200_MIL_MultiTask_imedslab_LCFLMB_MA_C0_Fo_lr1e-04_b16\
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CoralFocalLoss_MultiTask_MetricsBalanced \
  --predict_criteria Coral_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train19_classweight_norm_CE_all_128

# Retrain 17 with AVG loss
python inference.py \
  --current_ckpt model_checkpoints_20250929_225636_epoch200_MIL_MultiTask_imedslab_LCEoM_MA_C0_Fo_lr1e-04_b16 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv  \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train17_classweight_norm_CE_all

python inference.py \
  --current_ckpt model_checkpoints_20250930_012744_epoch200_MIL_MultiTask_imedslab_LCEoM_MA_C0_Fo_lr1e-04_b16 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv  \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train17_classweight_norm_CE_all


python inference.py \
  --current_ckpt model_checkpoints_20250929_223935_epoch200_MIL_MultiTask_imedslab_LCEoM_MKJ_C0_Fo_lr1e-04_b16 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note train17_classweight_norm_CE_kl_jsn

# Retrain to see the loss
python train.py \
  --model_type MILOrdinal_MultiTask_imedslab \
  --lossfcn_type CoralLoss_MultiTask \
  --predict_criteria Coral_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train19_classweight_norm_CE_all

python inference.py \
  --current_ckpt model_checkpoints_20250930_173341_epoch200_MILOrdinal_MultiTask_imedslab_LCLoM_MA_C0_Fo_lr1e-04_b16 \
  --model_type MILOrdinal_MultiTask_imedslab \
  --lossfcn_type CoralLoss_MultiTask \
  --predict_criteria Coral_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train19_classweight_norm_CE_all


# Reproduce
python train.py \
  --seed 42 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note final_reproduce_1

python inference.py \
  --current_ckpt model_checkpoints_20251006_001733_epoch200_MIL_MultiTask_imedslab_LCEoM_MKJ_C0_Fo_lr1e-04_b16 \
  --seed 42 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note final_reproduce_1

python train.py \
  --seed 0 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note final_reproduce_2

python inference.py \
  --current_ckpt model_checkpoints_20251006_001755_epoch200_MIL_MultiTask_imedslab_LCEoM_MKJ_C0_Fo_lr1e-04_b16 \
  --seed 0 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note final_reproduce_2

python train.py \
  --seed 2025 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note final_reproduce_3

python inference.py \
  --current_ckpt model_checkpoints_20251006_001804_epoch200_MIL_MultiTask_imedslab_LCEoM_MKJ_C0_Fo_lr1e-04_b16 \
  --seed 2025 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note final_reproduce_3

python train.py \
  --seed 2022 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note final_reproduce_4

python inference.py \
  --current_ckpt model_checkpoints_20251006_001819_epoch200_MIL_MultiTask_imedslab_LCEoM_MKJ_C0_Fo_lr1e-04_b16 \
  --seed 2022 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note final_reproduce_4

  python train.py \
  --seed 2024 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note final_reproduce_5

python inference.py \
  --current_ckpt model_checkpoints_20251006_001832_epoch200_MIL_MultiTask_imedslab_LCEoM_MKJ_C0_Fo_lr1e-04_b16 \
  --seed 2024 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type kl_jsn \
  --feedback_type off \
  --feedback_cam off \
  --note final_reproduce_5

# Reproduce again
python train.py \
  --seed 42 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note final_reproduce_1_again

python inference.py \
  --current_ckpt model_checkpoints_20251109_032115_epoch200_MIL_MultiTask_imedslab_LCEoM_MA_C0_Fo_lr1e-04_b16 \
  --seed 42 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note final_reproduce_1_again

python train.py \
  --seed 0 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note final_reproduce_2_again

python inference.py \
  --current_ckpt model_checkpoints_20251109_032130_epoch200_MIL_MultiTask_imedslab_LCEoM_MA_C0_Fo_lr1e-04_b16 \
  --seed 0 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note final_reproduce_2_again

python train.py \
  --seed 2024 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note final_reproduce_3_again

python inference.py \
  --current_ckpt model_checkpoints_20251109_032253_epoch200_MIL_MultiTask_imedslab_LCEoM_MA_C0_Fo_lr1e-04_b16 \
  --seed 2024 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note final_reproduce_3_again


python train.py \
  --seed 2022 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note final_reproduce_4_again

python inference.py \
  --current_ckpt model_checkpoints_20251109_032310_epoch200_MIL_MultiTask_imedslab_LCEoM_MA_C0_Fo_lr1e-04_b16 \
  --seed 2022 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note final_reproduce_4_again

python train.py \
  --seed 2025 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note final_reproduce_5_again

python inference.py \
  --current_ckpt model_checkpoints_20251109_032342_epoch200_MIL_MultiTask_imedslab_LCEoM_MA_C0_Fo_lr1e-04_b16 \
  --seed 2025 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note final_reproduce_5_again


python train_k_fold.py \
  --debug \
  --seed 2025 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note k_fold

python train_k_fold.py \
  --seed 42 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note k_fold_42

python inference_k_fold.py \
  --current_ckpt model_checkpoints_20251119_025306_epoch200_MIL_MultiTask_imedslab_LCEoM_MA_C0_Fo_lr1e-04_b16 \
  --seed 42 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note k_fold_42