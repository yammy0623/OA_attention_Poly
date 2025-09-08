import os
import time
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report, ConfusionMatrixDisplay
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from data_augmentation import CorrectBrightness, CorrectContrast, CorrectGamma
from dataset import KneeMILDataset, mil_collate_fn
from model import CompleteMILModel, CompleteMILOrdinalModel, CompleteMILOrdinal_MultiTask_Model
import argparse
import matplotlib.pyplot as plt
import wandb
import json
from datetime import datetime
import shutil
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, LayerCAM
from config import build_config  
from losses import CoralLossWeighted, CoralLossEffective, CoralFocalLoss, MultiTask_CoralFocalLoss, coral_predict, coral_multitask_predict  
from myutils import calculate_mean_std, compute_effective_class_weights
import torchvision.transforms as transforms
from enum import Enum

class Config:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)


# Get today’s date and time in YYYYMMDD_HHMM format
# weight*(patch + attention patch)
# NOW = datetime.now().strftime('%Y%m%d_%H%M%S')

# DEBUG_MODE=False
# WANDB= not DEBUG_MODE
# DATA_HALF=False

# # ---------------- Configuration ---------------- #
# # H5_FILE = os.path.join("model_checkpoints_tnc_final", "knee_patches_patient_grouped_16_100.h5")
# H5_FILE = "knee_patches_patient_grouped_16_100_all_feature.h5"
# PRE_CHECKPOINT_DIR = "model_checkpoints_tnc_final"
# CHECKPOINT_DIR = os.path.join("original_data", "V00", f"model_checkpoints_{NOW}_epoch200_finalckpt_100")
# MEAN_STD_FILE_PATH = os.path.join(CHECKPOINT_DIR, "mean_std_train_patches.npy")
# PRETRAINED_MODEL_PATH = os.path.join(PRE_CHECKPOINT_DIR, "best_model_val_kappa.pth")

# NUM_CLASSES = 5
# FEATURE_EXTRACTOR_OUT_DIM = 128
# AGGREGATION_TYPE = 'attention'
# LEARNING_RATE = 1e-4
# wd = 1e-4
# WEIGHT_DECAY = 1e-4
# BATCH_SIZE = 16
# NUM_EPOCHS = 200
# SEED = 42
# DEFAULT_MAX_PIXEL_VALUE = 65535.0

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NUM_WORKERS = 0
# PIN_MEMORY = DEVICE.type == 'cuda' and NUM_WORKERS > 0

    
def prepare_data(h5_file):
    with h5py.File(h5_file, 'r') as hf:
        base_ids = [pid.decode() for pid in hf['patient_ids_order'][:]]
        groups, grades = [], []
        for pid in base_ids:
            for side in ["_L","_R"]:
                g = pid + side
                if g in hf and hf[g]['kl_grade'][0] != -999 and hf[g]['patches'].shape[0] > 0:
                    groups.append(g)
                    grades.append(hf[g]['kl_grade'][0])
    return groups, grades


def create_transforms(mean, std):
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        CorrectBrightness(0.7,1.3),
        CorrectContrast(0.7,1.3),
        CorrectGamma(0.5,2.5,res=8),
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist())
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist())
    ])
    return train_transform, val_transform


def build_CAM_attention_tool(feedback_cam, model_org):
    """Helper: initialize GradCAM / CAM tool based on feedback_cam"""
    if model_org is None or feedback_cam == "original":
        return None

    target_layer = [model_org.patch_feature_extractor.conv_block3[0]]
    if feedback_cam == "GradCAM":
        return GradCAM(model=model_org.patch_feature_extractor, target_layers=target_layer)
    elif feedback_cam == "GradCAMPlusPlus":
        return GradCAMPlusPlus(model=model_org.patch_feature_extractor, target_layers=target_layer)
    elif feedback_cam == "ScoreCAM":
        return ScoreCAM(model=model_org.patch_feature_extractor, target_layers=target_layer)
    elif feedback_cam == "AblationCAM":
        return AblationCAM(model=model_org.patch_feature_extractor, target_layers=target_layer)
    elif feedback_cam == "LayerCAM":
        return LayerCAM(model=model_org.patch_feature_extractor, target_layers=target_layer)
    elif feedback_cam == "off":
        return None
    else:
        raise ValueError(f"Unknown feedback_cam: {feedback_cam}")
    

def run_epoch(loader, model, model_org, criterion, optimizer, device, is_training, config, desc=""):
    """
    Run one epoch of training/validation
    config: parsed argparse with flags like config.use_multitask, config.use_ordinal, config.training_type
    """
    model.train() if is_training else model.eval()
    total_loss, num_processed_samples = 0.0, 0

    # Prepare prediction containers
    if config.multitask_type == "off":
        all_preds, all_labels = [], []
    else:
        all_preds = {task: [] for task in config.OARSI_TASKS.keys()}
        all_labels = {task: [] for task in config.OARSI_TASKS.keys()}
        

    # Setup attention tool if applicable
    attention_tool = build_CAM_attention_tool(config.feedback_cam, model_org) if model_org else None
    if model_org:
        model_org.eval()

    progress_bar = tqdm(loader, desc=desc, leave=False)

    for list_of_patch_bags, labels_batch, group_name, list_of_features in progress_bar:
        if not list_of_patch_bags:
            continue

        # Move valid bags + features
        moved_bags, moved_features, valid_indices = [], [], []
        for i, bag in enumerate(list_of_patch_bags):
            if bag.nelement() > 0:
                moved_bags.append(bag.to(device, non_blocking=True))
                moved_features.append(list_of_features[i][0].to(device, non_blocking=True))
                valid_indices.append(i)

        if not moved_bags:
            continue

        labels_batch = labels_batch[valid_indices].to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            # Forward pass
            if config.feedback_type == "off":
                outputs, _, _, _ = model(moved_bags)
            else:
                outputs, _, _, _ = model(moved_bags, model_org, attention_tool)


            # Target handling
            if config.multitask_type == "off":
                loss = criterion(outputs, labels_batch)
                
            else:
                if config.multitask_type == "all":
                    targets = {
                        "kl":   labels_batch,
                        "jsnm": torch.tensor([f[0] for f in moved_features], device=device),
                        "jsnl": torch.tensor([f[1] for f in moved_features], device=device),
                        "osfm": torch.tensor([f[2] for f in moved_features], device=device),
                        "ostm": torch.tensor([f[3] for f in moved_features], device=device),
                        "ostl": torch.tensor([f[4] for f in moved_features], device=device),
                        "osfl": torch.tensor([f[5] for f in moved_features], device=device),
                    }
                    # Replace -999 with 0
                    for k, v in targets.items():
                        targets[k] = torch.where(v == -999, torch.tensor(0, device=device), v)
                elif config.multitask_type == "kl_jsn":
                    targets = {
                        "kl":   labels_batch,
                        "jsnm": torch.tensor([f[0] for f in moved_features], device=device),
                        "jsnl": torch.tensor([f[1] for f in moved_features], device=device),
                    }
                # print("outputs", outputs)
                # print("targets", targets)
                # print("criterion", criterion)
                loss, loss_dict = criterion(outputs, targets)


            # Backward
            if is_training:
                loss.backward()
                optimizer.step()

        # Update running loss
        total_loss += loss.item() * labels_batch.size(0)
        num_processed_samples += labels_batch.size(0)

        # Predictions
        if config.multitask_type == "off":
            if config.predict_criteria == "Coral":
                predicted = coral_predict(outputs)
            elif config.predict_criteria == "Max":
                _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
        else:
            if config.predict_criteria == "Coral_Multitask":
                predicted = coral_multitask_predict(outputs)
            for task in config.OARSI_TASKS.keys():
                all_preds[task].extend(predicted[task].cpu().numpy())
                all_labels[task].extend(targets[task].cpu().numpy())

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / num_processed_samples if num_processed_samples > 0 else 0
    return avg_loss, all_labels, all_preds, num_processed_samples



def compute_metrics(multitask_type, labels, preds):
    metrics = {}
    if multitask_type == "off":
        metrics["kl"] = {
            "acc": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average='weighted', zero_division=0),
            "kappa": cohen_kappa_score(labels, preds, weights="quadratic")
        }
    else:
        metrics = {}
        for task in labels.keys():
            metrics[task] = {
                "acc": accuracy_score(labels[task], preds[task]),
                "f1": f1_score(labels[task], preds[task], average='weighted', zero_division=0),
                "kappa": cohen_kappa_score(labels[task], preds[task], weights="quadratic")
            }
    return metrics
    
def save_checkpoint(model, checkpoint_dir, name):
    path = os.path.join(checkpoint_dir, name)
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint: {path}")

def get_criterion(lossfcn_type, class_weights_tensor, oai_task_num_classes=None):
    if lossfcn_type == "CoralLossWeighted":
        return CoralLossWeighted(class_weights=class_weights_tensor)
    elif lossfcn_type == "CoralFocalLoss":
        return CoralFocalLoss(class_weights=class_weights_tensor, gamma=2.0, alpha=0.25)
    elif lossfcn_type == "CoralLossEffective":
        return CoralLossEffective(threshold_weights=class_weights_tensor)
    elif lossfcn_type == "CoralFocalLoss_MultiTask":
        return MultiTask_CoralFocalLoss(oai_task_num_classes, is_learn_task_weights=True, class_weights=class_weights_tensor)
    elif lossfcn_type == "CrossEntropy":
        return nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        print("No criterion")
    
# Compute class weights
def compute_class_weights(classweight_type, counts, beta=0.9999, device="cuda"):
    if classweight_type== "effective":  # effective class weights
        weights = compute_effective_class_weights(counts, num_classes=len(counts), beta=beta)
        weights = np.array(weights)
        # Optional scaling for specific classes
        weights[1, :] = weights[1, :] * 2.0
    else:  # inverse frequency weights
        weights = 1.0 / (counts + 1e-6)  # Add epsilon
        weights = weights / np.sum(weights) * len(counts)  # Normalize to num_classes

    return torch.tensor(weights, dtype=torch.float).to(device)


def log_metrics(metrics_dict, prefix, epoch, use_wandb=True):
    """
    metrics_dict can be multi-task (dict of dicts) or single dict
    """
    if isinstance(metrics_dict[list(metrics_dict.keys())[0]], dict):
        # multi-task
        for task, m in metrics_dict.items():
            line = f"[{prefix}] {task} - Acc: {m['acc']:.4f}, F1: {m['f1']:.4f}, Kappa: {m['kappa']:.4f}"
            print(line)
            if use_wandb:
                wandb.log({
                    f"{prefix}/{task}_accuracy": m['acc'],
                    f"{prefix}/{task}_f1_weighted": m['f1'],
                    f"{prefix}/{task}_kappa": m['kappa'],
                    "epoch": epoch
                })
        # aggregate across tasks
        agg = {
            'acc': np.mean([m['acc'] for m in metrics_dict.values()]),
            'f1': np.mean([m['f1'] for m in metrics_dict.values()]),
            'kappa': np.mean([m['kappa'] for m in metrics_dict.values()])
        }
    else:
        # single task
        m = metrics_dict
        line = f"[{prefix}] Acc: {m['acc']:.4f}, F1: {m['f1']:.4f}, Kappa: {m['kappa']:.4f}"
        print(line)
        if use_wandb:
            wandb.log({
                f"{prefix}/accuracy": m['acc'],
                f"{prefix}/f1_weighted": m['f1'],
                f"{prefix}/kappa": m['kappa'],
                "epoch": epoch
            })
        agg = m
    return agg

def save_best_models(model, metrics_dict, best_metrics, best_mean_metrics, checkpoint_dir):
    """
    metrics_dict: single or multi-task metrics
    best_metrics: dict of best values
    Updates best_metrics and saves checkpoint if new best
    """
    if isinstance(metrics_dict[list(metrics_dict.keys())[0]], dict):
        # Multi-task
        kl_metrics = metrics_dict.get("kl", {})
        avg_metrics = {key: np.mean([m[key] for m in metrics_dict.values()]) for key in ['acc','f1','kappa']}

        for key in ['acc','f1','kappa']:
            # KL-best
            kl_val = kl_metrics.get(key, -np.inf)
            if kl_val > best_metrics.get(f"kl_{key}", -np.inf):
                best_metrics[f"kl_{key}"] = kl_val
                path = os.path.join(checkpoint_dir, f"best_model_kl_{key}.pth")
                torch.save(model.state_dict(), path)
                print(f"  Saved new best KL {key} model ({key}: {kl_val:.4f})")

            # AVG-best
            avg_val = avg_metrics[key]
            if avg_val > best_mean_metrics.get(f"avg_{key}", -np.inf):
                best_mean_metrics[f"avg_{key}"] = avg_val
                path = os.path.join(checkpoint_dir, f"best_model_avg_{key}.pth")
                torch.save(model.state_dict(), path)
                print(f"  Saved new best AVG {key} model ({key}: {avg_val:.4f})")

        print("Average metrics across all tasks:")
    else:
        for key in ['acc','f1','kappa']:
            if metrics_dict[key] > best_metrics.get(key, -np.inf):
                best_metrics[key] = metrics_dict[key]
                path = os.path.join(checkpoint_dir, f"best_model_{key}.pth")
                torch.save(model.state_dict(), path)
                print(f"  Saved new best {key} model ({key}: {metrics_dict[key]:.4f})")

def main(config):
    # ----------------- Setup ----------------- #
    # Copy source files for reproducibility
    files_to_copy = ["train.py", "model.py", "dataset.py", "data_augmentation.py", "losses.py", "utils.py", "config.py"]
    if not config.DEBUG_MODE:
        for file in files_to_copy:
            if os.path.exists(file):
                shutil.copy(file, config.CHECKPOINT_DIR)
                print(f"Copied {file} to {config.CHECKPOINT_DIR}")
            else:
                print(f"WARNING: {file} not found and was not copied.")

        # Initialize wandb
        wandb.init(
            project="Knee_OA_MIL",
            name=config.run_name,
            config=vars(config),
            tags=[
                f"lr{config.LEARNING_RATE:.0e}",
                f"b{config.BATCH_SIZE}",
                f"s{config.SEED}",
                f"e{config.NUM_EPOCHS}"
            ],
        )
    # ----------------- Data ----------------- #
    groups, grades = prepare_data(config.H5_FILE)
    print(f"Total valid samples: {len(groups)}")

    # Train/val/test split
    train_val, test, train_val_grades, _ = train_test_split(
        np.array(groups), np.array(grades), test_size=0.2, stratify=grades, random_state=config.SEED
    )
    train, val, _, _ = train_test_split(
        train_val, train_val_grades, test_size=0.25, stratify=train_val_grades, random_state=config.SEED
    )

    train_pids, val_pids, test_pids = train.tolist(), val.tolist(), test.tolist()
    if "9491446_R" in test_pids:  # remove bad image
        test_pids.remove("9491446_R")

    print(f"Training samples: {len(train_pids)}, Validation: {len(val_pids)}, Testing: {len(test_pids)}")

    # Compute or load mean/std: normalizing input data before feeding it into the model.
    if os.path.exists(config.MEAN_STD_FILE_PATH):
        mean, std = np.load(config.MEAN_STD_FILE_PATH)
    else:
        mean, std = calculate_mean_std(config.H5_FILE, train, config.MEAN_STD_FILE_PATH, config.DEFAULT_MAX_PIXEL_VALUE)
    print(f"Mean: {mean}, Std: {std}")

    train_transform, val_transform = create_transforms(mean, std)

    # Handle DATA_HALF option
    if config.DATA_HALF:
        train_pids, val_pids, test_pids = train_pids[:len(train_pids)//2], val_pids[:len(val_pids)//2], test_pids[:len(test_pids)//2]
        print(f"Using half dataset: train {len(train_pids)}, val {len(val_pids)}, test {len(test_pids)}")

    # Datasets and loaders
    train_ds = KneeMILDataset(config.H5_FILE, train_pids, transform=train_transform)
    val_ds = KneeMILDataset(config.H5_FILE, val_pids, transform=val_transform)
    test_ds = KneeMILDataset(config.H5_FILE, test_pids, transform=val_transform)

    train_loader = DataLoader(train_ds, config.BATCH_SIZE, True, collate_fn=mil_collate_fn,
                              num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    val_loader = DataLoader(val_ds, config.BATCH_SIZE, False, collate_fn=mil_collate_fn,
                            num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    test_loader = DataLoader(test_ds, config.BATCH_SIZE, False, collate_fn=mil_collate_fn,
                             num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

    # ----------------- Model ----------------- #
    if config.model_type == "MTLOrdinal":  # ordinal model
        model = CompleteMILOrdinalModel(config.FEATURE_EXTRACTOR_OUT_DIM, config.KL_NUM_CLASSES, config.AGGREGATION_TYPE).to(config.DEVICE)
    elif config.model_type == "MTLOrdinal_MultiTask":  # multitask
        model = CompleteMILOrdinal_MultiTask_Model(config.FEATURE_EXTRACTOR_OUT_DIM,
                                                   config.OARSI_TASKS,
                                                   config.AGGREGATION_TYPE).to(config.DEVICE)
    elif config.model_type == "MTL": 
        model = CompleteMILModel(config.FEATURE_EXTRACTOR_OUT_DIM,
                                     config.KL_NUM_CLASSES,
                                     config.AGGREGATION_TYPE).to(config.DEVICE)
    # else:
    #     model = CompleteMILCamModel_Attention_feedback(config.FEATURE_EXTRACTOR_OUT_DIM,
    #                                                    config.NUM_CLASSES,
    #                                                    config.training_type,
    #                                                    config.feedback_type,
    #                                                    config.AGGREGATION_TYPE,
    #                                                    num_features=config.num_features).to(config.DEVICE)
    if config.feedback_type == "off":
        model_org = None
    else:
        model_org = CompleteMILModel(config.FEATURE_EXTRACTOR_OUT_DIM,
                                     config.KL_NUM_CLASSES,
                                     config.AGGREGATION_TYPE).to(config.DEVICE)
        model_org.load_state_dict(torch.load(config.PRETRAINED_MODEL_PATH, map_location=config.DEVICE))

    # ----------------- Loss & Optimizer ----------------- #
    train_kl_grades = []
    with h5py.File(config.H5_FILE, 'r') as hf:
        for group_name in train_ds.sample_group_names:
            train_kl_grades.append(hf[group_name]['kl_grade'][0])
    class_counts = np.bincount(train_kl_grades, minlength=config.KL_NUM_CLASSES)
    print(f"Class counts in training set: {class_counts}")

    class_weights_tensor = compute_class_weights(config.classweight_type, class_counts, device=config.DEVICE)
    print(f"Using class weights: {class_weights_tensor}")

    criterion = get_criterion(config.lossfcn_type, class_weights_tensor, config.OARSI_TASKS)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    # ----------------- Training Loop ----------------- #
    best_metrics = {'kl_acc':0, 'kl_f1':0, 'kl_kappa':-1}  # for example
    best_mean_metrics = {'avg_acc':0, 'avg_f1':0, 'avg_kappa':-1}

    for epoch in range(config.NUM_EPOCHS):
        torch.cuda.empty_cache()
        epoch_num = epoch + 1

        # Train
        train_loss, train_labels, train_preds, processed_train_samples = run_epoch(
            train_loader, model, model_org, criterion, optimizer, config.DEVICE,
            is_training=True, config=config,
            desc=f"Epoch {epoch_num}/{config.NUM_EPOCHS} [Train]"
        )
        if processed_train_samples > 0:
            train_metrics = compute_metrics(config.multitask_type, train_labels, train_preds)
            log_metrics(train_metrics, "Train", epoch_num, use_wandb=config.WANDB)

        # Validate
        val_loss, val_labels, val_preds, processed_val_samples = run_epoch(
            val_loader, model, model_org, criterion, None, config.DEVICE,
            is_training=False, config=config,
            desc=f"Epoch {epoch_num}/{config.NUM_EPOCHS} [Val]"
        )
        scheduler.step(val_loss)

        if processed_val_samples > 0:
            val_metrics = compute_metrics(config.multitask_type, val_labels, val_preds)
            log_metrics(val_metrics, "Val", epoch_num, use_wandb=config.WANDB)

        # Save best
        kl_metrics = val_metrics.get("kl", {})
        for key in ['acc','f1','kappa']:
            kl_val = kl_metrics.get(key, -np.inf)
            if kl_val > best_metrics.get(f"kl_{key}", -np.inf):
                best_metrics[f"kl_{key}"] = kl_val
                path = os.path.join(config.CHECKPOINT_DIR, f"best_model_kl_{key}.pth")
                torch.save(model.state_dict(), path)
                print(f"  Saved new best KL {key} model ({key}: {kl_val:.4f})")
            
            if not config.multitask_type == "off":
                avg_metrics = {key: np.mean([val_metrics[task][key] for task in val_metrics])}
                # AVG-best
                avg_val = avg_metrics[key]
                if avg_val > best_mean_metrics.get(f"avg_{key}", -np.inf):
                    best_mean_metrics[f"avg_{key}"] = avg_val
                    path = os.path.join(config.CHECKPOINT_DIR, f"best_model_avg_{key}.pth")
                    torch.save(model.state_dict(), path)
                    print(f"  Saved new best AVG {key} model ({key}: {avg_val:.4f})")

            print("Average metrics across all tasks:")

    print("Training finished.")
    if config.WANDB:
        wandb.finish()


if __name__ == "__main__":
    from config import build_config
    config_dict = build_config()
    cfg = Config(config_dict)
    main(cfg)