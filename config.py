import os
import torch
import argparse
from datetime import datetime

# ---------------- Default Configuration ---------------- #
NOW = datetime.now().strftime('%Y%m%d_%H%M%S')

DEBUG_MODE = False
WANDB = not DEBUG_MODE
DATA_HALF = False

# Dataset / Checkpoints
DEFAULT_H5_FILE = "knee_patches_patient_grouped_16_100_all_feature.h5"
DEFAULT_PRE_CKPT_DIR = "model_checkpoints_tnc_final"
DEFAULT_PRETRAINED_MODEL = os.path.join(DEFAULT_PRE_CKPT_DIR, "best_model_val_kappa.pth")

# Training hyperparameters
KL_NUM_CLASSES = 5
OARSI_TASKS = {
            "jsnm": 4,  # 0–3 ordinal
            "jsnl": 4,  # 0–3 ordinal
            "osfm": 4,  # 0–3 ordinal
            "ostm": 4,  # 0–3 ordinal
            "ostl": 4,  # 0–3 ordinal
            "osfl": 4,  # 0–3 ordinal
        }
NUM_FEATURES = len(OARSI_TASKS)

FEATURE_EXTRACTOR_OUT_DIM = 128
AGGREGATION_TYPE = "attention"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 200
SEED = 42
DEFAULT_MAX_PIXEL_VALUE = 65535.0

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
PIN_MEMORY = DEVICE.type == "cuda" and NUM_WORKERS > 0


# ---------------- Argument Parser ---------------- #
def get_args():
    parser = argparse.ArgumentParser()

    # Experiment setup
    parser.add_argument(
        "--training_type",
        type=str,
        default="original",
        choices=["original", "GradCAM", "GradCAMPlusPlus", "ScoreCAM", "AblationCAM", "LayerCAM"]
    )
    parser.add_argument(
        "--pre_ckpt",
        type=str,
        default=DEFAULT_PRE_CKPT_DIR,
        help="Path to pretrained checkpoint dir"
    )
    parser.add_argument(
        "--feedback_type",
        type=int,
        default=1,
        choices=list(range(1, 15)),
        help="Feedback strategy (1–14)"
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Additional note for experiment naming"
    )

    parser.add_argument("--use_baseline", action="store_true", help="Use baseline MIL model")
    parser.add_argument("--use_ordinal", action="store_true", help="Use ordinal MIL loss")
    parser.add_argument("--use_weighted_cam", action="store_true", help="Use weighted CAM mechanism")
    parser.add_argument("--use_multitask", action="store_true", help="Enable multitask learning")

    parser.add_argument(
        "--num_features",
        type=int,
        default=6,
        help="Number of OARSI features used in multitask mode"
    )

    args = parser.parse_args()
    return args


# ---------------- Build Final Config ---------------- #
def build_config():
    args = get_args()

    # number of features (you can adjust this logic if dynamic)
    num_features = 6

    # dataset info
    data_part = "halfdata" if DATA_HALF else "wholedata"

    # timestamp + run_name
    timestamp = datetime.now().strftime('%m%d_%H%M')
    run_name = (
        f"{args.training_type}_lr{LEARNING_RATE:.0e}_b{BATCH_SIZE}_{timestamp}"
        f"_feedback_{args.feedback_type}_{data_part}_{args.note}_feat_{num_features}"
    )

    # checkpoint dir
    checkpoint_dir = os.path.join(
        "original_data",
        "V00",
        f"model_checkpoints_{NOW}_epoch{NUM_EPOCHS}_finalckpt_100_feedback_{args.feedback_type}_{args.note}_feat_{num_features}"
    )

    config = {
        # experiment setup
        "NOW": NOW,
        "DEBUG_MODE": DEBUG_MODE,
        "WANDB": WANDB,
        "DATA_HALF": DATA_HALF,

        # dataset paths
        "H5_FILE": DEFAULT_H5_FILE,
        "PRE_CHECKPOINT_DIR": args.pre_ckpt,
        "CHECKPOINT_DIR": checkpoint_dir,
        "MEAN_STD_FILE_PATH": os.path.join(checkpoint_dir, "mean_std_train_patches.npy"),
        "PRETRAINED_MODEL_PATH": DEFAULT_PRETRAINED_MODEL,

        # hyperparameters
        "FEATURE_EXTRACTOR_OUT_DIM": FEATURE_EXTRACTOR_OUT_DIM,
        "AGGREGATION_TYPE": AGGREGATION_TYPE,
        "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_EPOCHS": NUM_EPOCHS,
        "SEED": SEED,
        "DEFAULT_MAX_PIXEL_VALUE": DEFAULT_MAX_PIXEL_VALUE,
        "NUM_FEATURES": num_features,
        "DATA_PART": data_part,

        # tasks
        "KL_NUM_CLASSES": KL_NUM_CLASSES,
        "OARSI_TASKS": OARSI_TASKS,
        "NUM_FEATURES": NUM_FEATURES,

        # Ablation study
        "baseline": args.use_baseline,
        "weighted_cam": args.use_weighted_cam,
        "ordinal": args.use_ordinal,
        "multitask": args.use_multitask,

        # device
        "DEVICE": DEVICE,
        "NUM_WORKERS": NUM_WORKERS,
        "PIN_MEMORY": PIN_MEMORY,

        # args
        "training_type": args.training_type,
        "feedback_type": args.feedback_type,
        "note": args.note,

        # info
        "timestamp": timestamp,
        "run_name": run_name,
    }

    # make dirs if needed
    os.makedirs(config["CHECKPOINT_DIR"], exist_ok=True)

    return config
