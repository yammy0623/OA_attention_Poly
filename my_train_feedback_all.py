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
from model import CompleteMILModel, CompleteMILCamModel, CompleteMILCamModel_Attention_feedback, CompleteMILOrdinalModel, CompleteMILOrdinal_MultiTask_Model
import argparse
import matplotlib.pyplot as plt
import wandb
import json
from datetime import datetime
import shutil
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, LayerCAM
    

# Get today’s date and time in YYYYMMDD_HHMM format
# weight*(patch + attention patch)
NOW = datetime.now().strftime('%Y%m%d_%H%M%S')

DEBUG_MODE=False
WANDB= not DEBUG_MODE
DATA_HALF=False

# ---------------- Configuration ---------------- #
# H5_FILE = os.path.join("model_checkpoints_tnc_final", "knee_patches_patient_grouped_16_100.h5")
H5_FILE = "knee_patches_patient_grouped_16_100_all_feature.h5"
PRE_CHECKPOINT_DIR = "model_checkpoints_tnc_final"
CHECKPOINT_DIR = os.path.join("original_data", "V00", f"model_checkpoints_{NOW}_epoch200_finalckpt_100")
MEAN_STD_FILE_PATH = os.path.join(CHECKPOINT_DIR, "mean_std_train_patches.npy")
PRETRAINED_MODEL_PATH = os.path.join(PRE_CHECKPOINT_DIR, "best_model_val_kappa.pth")

NUM_CLASSES = 5
FEATURE_EXTRACTOR_OUT_DIM = 128
AGGREGATION_TYPE = 'attention'
LEARNING_RATE = 1e-4
wd = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 200
SEED = 42
DEFAULT_MAX_PIXEL_VALUE = 65535.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
PIN_MEMORY = DEVICE.type == 'cuda' and NUM_WORKERS > 0

import torch.nn.functional as F

class CoralLossWeighted(nn.Module):
    """
    CORAL loss with class weights to handle imbalanced data
    """
    def __init__(self, class_weights=None):
        """
        Args:
            class_weights: Tensor of shape (num_classes,)
                           對應到每個 class 的權重，例如 [w0, w1, w2, w3, w4]
        """
        super(CoralLossWeighted, self).__init__()
        self.class_weights = class_weights

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, K-1)
            targets: (batch_size,)
        """
        batch_size, num_classes_minus1 = logits.shape
        prob = torch.sigmoid(logits)

        # 建立 target matrix (binary)
        target_matrix = torch.zeros((batch_size, num_classes_minus1), device=logits.device)
        for i in range(batch_size):
            target_matrix[i, :targets[i]] = 1

        # 如果有 class weight
        if self.class_weights is not None:
            # 依照每個樣本的 true label 取對應權重
            sample_weights = self.class_weights[targets]   # (batch_size,)
            # 擴展到 (batch_size, K-1)，讓每個 cutpoint 都能乘上
            sample_weights = sample_weights.unsqueeze(1).expand_as(prob)
        else:
            sample_weights = torch.ones_like(prob)

        # Binary cross-entropy with weights
        loss = F.binary_cross_entropy(prob, target_matrix, weight=sample_weights, reduction="mean")

        return loss


class CoralLossEffective(nn.Module):
    """
    CORAL loss with per-threshold effective number weighting
    """
    def __init__(self, threshold_weights=None):
        """
        Args:
            threshold_weights: Tensor of shape (K-1, 2),
                               每個 threshold 的 [w_neg, w_pos]
        """
        super(CoralLossEffective, self).__init__()
        self.threshold_weights = threshold_weights  # (K-1, 2)

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, K-1)
            targets: (batch_size,)
        """
        batch_size, num_classes_minus1 = logits.shape
        prob = torch.sigmoid(logits)

        # target_matrix: (batch_size, K-1)
        target_matrix = torch.zeros((batch_size, num_classes_minus1), device=logits.device)
        for i in range(batch_size):
            target_matrix[i, :targets[i]] = 1

        # per-threshold BCE with weights
        loss_matrix = torch.zeros_like(prob)
        for k in range(num_classes_minus1):
            w_neg, w_pos = self.threshold_weights[k]

            # BCE 分開寫正負
            loss_matrix[:, k] = - (
                w_pos * target_matrix[:, k] * torch.log(prob[:, k] + 1e-8) +
                w_neg * (1 - target_matrix[:, k]) * torch.log(1 - prob[:, k] + 1e-8)
            )

        return loss_matrix.mean()
    
class CoralFocalLoss(nn.Module):
    """
    Focal-CORAL loss with optional class weights for imbalanced data.
    """
    def __init__(self, class_weights=None, gamma=2.0, alpha=0.25):
        """
        Args:
            class_weights: Tensor of shape (num_classes,), optional class-level weight
            gamma: focusing parameter, typical value 2.0
            alpha: balance parameter between positive/negative samples
        """
        super(CoralFocalLoss, self).__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape (batch_size, K-1), raw outputs
            targets: Tensor of shape (batch_size,), true labels (0~K-1)
        """
        batch_size, num_classes_minus1 = logits.shape
        prob = torch.sigmoid(logits)

        # 建立 target matrix (binary)
        target_matrix = torch.zeros((batch_size, num_classes_minus1), device=logits.device)
        for i in range(batch_size):
            target_matrix[i, :targets[i]] = 1

        # 計算 p_t (對應正負樣本)
        pt = torch.where(target_matrix == 1, prob, 1 - prob)

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = torch.where(target_matrix == 1, self.alpha, 1 - self.alpha)

        # Focal-CORAL loss matrix
        loss_matrix = - alpha_t * focal_weight * (
            target_matrix * torch.log(prob + 1e-8) +
            (1 - target_matrix) * torch.log(1 - prob + 1e-8)
        )

        # 加上 class weight
        if self.class_weights is not None:
            sample_weights = self.class_weights[targets]  # (batch,)
            sample_weights = sample_weights.unsqueeze(1).expand_as(loss_matrix)
            loss_matrix = loss_matrix * sample_weights

        return loss_matrix.mean()

class MultiTask_CoralFocalLoss(nn.Module):

    """
    Focal-CORAL loss with optional class weights for imbalanced data.
    """
    def __init__(self, task_num_classes, is_learn_task_weights, class_weights=None, gamma=2.0, alpha=0.25):
        """
        Args:
            class_weights: Tensor of shape (num_classes,), optional class-level weight
            gamma: focusing parameter, typical value 2.0
            alpha: balance parameter between positive/negative samples
            learn_task_weights: if True, learn uncertainty-based weights for tasks
        """
        super(MultiTask_CoralFocalLoss, self).__init__()
        self.task_num_classes = task_num_classes
        self.class_weights = class_weights
        self.gamma = gamma
        self.alpha = alpha

        if is_learn_task_weights:
            self.log_vars = nn.ParameterDict({
                t: nn.Parameter(torch.zeros(1)) for t in task_num_classes
            })
        else:
            self.log_vars = None

    def coral_focal_loss(self, logits, targets, kl_logits):
        """
        Args:
            logits: Tensor of shape (batch_size, K-1), raw outputs
            targets: Tensor of shape (batch_size,), true labels (0~K-1)
        """
        batch_size, num_classes_minus1 = logits.shape
        prob = torch.sigmoid(logits)
        if self.log_vars:
            self.log_vars.to(logits.device)

        # 建立 target matrix (binary)
        target_matrix = torch.zeros((batch_size, num_classes_minus1), device=logits.device)
        for i in range(batch_size):
            target_matrix[i, :int(targets[i])] = 1

        # 計算 p_t (對應正負樣本)
        pt = torch.where(target_matrix == 1, prob, 1 - prob)

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = torch.where(target_matrix == 1, self.alpha, 1 - self.alpha)

        # Focal-CORAL loss matrix
        loss_matrix = - alpha_t * focal_weight * (
            target_matrix * torch.log(prob + 1e-8) +
            (1 - target_matrix) * torch.log(1 - prob + 1e-8)
        )

        # 加上 class weight
        if self.class_weights is not None:
            sample_weights = self.class_weights[kl_logits]  # (batch,)
            sample_weights = sample_weights.unsqueeze(1).expand_as(loss_matrix)
            loss_matrix = loss_matrix * sample_weights

        return loss_matrix.mean()

    def forward(self, outputs, targets):
        """
        Args:

            outputs:{
                "kl":
                "jsnm":
                "jsnl":
            }

            targets: dict of labels
        """
        total_loss = 0
        loss_dict = {}

        # print("Outputs keys:", outputs.keys())
        for task, preds in outputs.items():
            if task not in targets:
                continue

            l = self.coral_focal_loss(
                preds,
                targets[task],
                kl_logits=targets["kl"]
            )

            if self.log_vars is not None and task in self.log_vars:
                w = torch.exp(-self.log_vars[task].to(l.device))
                total_loss += w * l + self.log_vars[task]
            else:
                total_loss += l

            loss_dict[task] = l.item()

        return total_loss, loss_dict

def coral_predict(logits):
    """
    logits: (batch_size, K-1)
    return: predicted class (batch_size,)
    """
    prob = torch.sigmoid(logits)   # (batch, K-1)
    # 檢查從左到右哪個 cutpoint 變成 <0.5
    preds = torch.sum(prob > 0.5, dim=1)
    return preds

def coral_multitask_predict(outputs):
    """
    outputs: dict of task_name -> logits
    returns: dict of task_name -> predicted labels
    """
    preds = {}
    for task, logits in outputs.items():
        preds[task] = coral_predict(logits)
    return preds

def compute_effective_class_weights(labels, num_classes, beta=0.9999):
    """
    計算 CORAL loss 每個 threshold 的 effective class weight
    
    Args:
        labels: (N,) tensor，包含 ordinal label，例如 [0,1,2,3,4]
        num_classes: 總類別數 (K)，例如 5
        beta: smoothing 參數，越接近1，越強調小樣本類別
    
    Returns:
        weights_per_threshold: list of (pos_weight, neg_weight) for each threshold
                               長度為 K-1
    """
    # print("Computing effective class weights for ", labels)
    N = labels.sum()
    weights_per_threshold = []

    for k in range(num_classes - 1):  # K-1 個 threshold
        # 定義正負樣本
        pos_idx = labels[k+1:].sum()
        neg_idx = N - pos_idx
        # print("pos_idx :", pos_idx )
        # print("neg_idx :", neg_idx )

        n_pos = np.sum(pos_idx)
        n_neg = np.sum(neg_idx)

        # 避免除零
        n_pos = max(1, n_pos)
        n_neg = max(1, n_neg)

        # effective number (Cui et al. 2019)
        eff_pos = (1 - beta) / (1 - beta**n_pos)
        eff_neg = (1 - beta) / (1 - beta**n_neg)

        # 取倒數當 weight
        w_pos = 1.0 / eff_pos
        w_neg = 1.0 / eff_neg

        # w_pos = w_pos ** 2
        # w_neg = w_neg ** 2

        # normalize，避免 scale 差太多
        s = w_pos + w_neg
        w_pos /= s
        w_neg /= s

        weights_per_threshold.append((w_pos, w_neg))

    return weights_per_threshold
# ---------------- Utilities ---------------- #
def calculate_mean_std(h5_file, sample_groups, save_path):
    num_channels = 1
    channel_sum = np.zeros(num_channels)
    channel_sum_sq = np.zeros(num_channels)
    total_pixel_count = 0

    with h5py.File(h5_file, 'r') as hf:
        for group_name in tqdm(sample_groups, desc="Calculating Mean/Std"):
            patches = hf[group_name]['patches'][:]
            if patches.size == 0:
                continue
            patches = patches.astype(np.float64) / DEFAULT_MAX_PIXEL_VALUE
            num_patches, H, W, _ = patches.shape
            current_pixels = num_patches * H * W
            total_pixel_count += current_pixels
            channel_sum += np.sum(patches, axis=(0, 1, 2))
            channel_sum_sq += np.sum(patches ** 2, axis=(0, 1, 2))

    mean = channel_sum / total_pixel_count
    variance = (channel_sum_sq / total_pixel_count) - mean ** 2
    std = np.sqrt(np.maximum(variance, 1e-7))
    np.save(save_path, [mean, std])
    return mean, std


def run_epoch(loader, model, model_org, criterion, optimizer, device, is_training, training_type, feedback_type, oai_task_num_classes=None, desc=""):
    gradcam_type = training_type
    model.train() if is_training else model.eval()
    total_loss, all_preds, all_labels, num_processed_samples = 0.0, [], [], 0
    if feedback_type in [14]:
        all_preds = {task: [] for task in oai_task_num_classes.keys()}
        all_labels = {task: [] for task in oai_task_num_classes.keys()}
    progress_bar = tqdm(loader, desc=desc, leave=False)


    if model_org:
        model_org.eval()

        target_layer = [model_org.patch_feature_extractor.conv_block3[0]]
        if gradcam_type == "GradCAM":
            attention_tool = GradCAM(
            model=model_org.patch_feature_extractor,     
            target_layers=target_layer,
            )
        elif gradcam_type == "GradCAMPlusPlus":
            attention_tool = GradCAMPlusPlus(
            model=model_org.patch_feature_extractor,   
            target_layers=target_layer,
            )
        elif gradcam_type == "ScoreCAM":  # should close the tqdm
            attention_tool = ScoreCAM(
            model=model_org.patch_feature_extractor,   
            target_layers=target_layer,
            )
        elif gradcam_type == "AblationCAM": # should close the tqdm
            attention_tool = AblationCAM(
            model=model_org.patch_feature_extractor,       
            target_layers=target_layer,
            )
        elif gradcam_type == "LayerCAM":
            attention_tool = LayerCAM(
            model=model_org.patch_feature_extractor,        
            target_layers=target_layer,
            )
        elif gradcam_type == "original":
            attention_tool = None
        else:
            print("Warning: No model")

    for list_of_patch_bags, labels_batch, group_name, list_of_features in progress_bar:
        if list_of_patch_bags is None or not list_of_patch_bags:
            # This might happen if collate_fn decides a batch is entirely invalid
            print(f"Warning: Skipped an entirely invalid batch in {desc}.")
            continue
        
        moved_list_of_patch_bags = []
        valid_indices_in_batch = [] # Keep track of which original labels correspond to moved bags
        moved_list_of_features = []

        for i, bag in enumerate(list_of_patch_bags):
            if bag.nelement() > 0:  # Check if bag is not empty
                moved_list_of_patch_bags.append(bag.to(DEVICE, non_blocking=PIN_MEMORY))
                valid_indices_in_batch.append(i)
        
        for i in valid_indices_in_batch:
                # print("feature: ", list_of_features[i][0][0])
                # feat = list_of_features[i][0][0].unsqueeze(0)
                # feat = list_of_features[i][0][:2]
                feat = list_of_features[i][0]
                # print(feat)
                # feat = list_of_features[i]
                moved_list_of_features.append(feat.to(DEVICE, non_blocking=PIN_MEMORY))

        
        # moved_list_of_patch_bags = torch.stack(moved_list_of_patch_bags).to(DEVICE)

        if not moved_list_of_patch_bags: # If all bags in this batch were empty after filtering
            # print(f"Warning: All bags in current batch for {epoch_desc} were empty. Skipping.")
            continue

        # Select labels corresponding to the valid bags
        labels_batch = labels_batch[valid_indices_in_batch].to(DEVICE, non_blocking=PIN_MEMORY)

        if is_training:
            optimizer.zero_grad()


        with torch.set_grad_enabled(is_training): # Context manager for gradients
            

            if feedback_type in [14]:
                outputs = model(moved_list_of_patch_bags) # Model takes the list of bags
                if outputs["kl"].shape[0] != labels_batch.shape[0]:
                    print(f"Shape mismatch in {desc}! Outputs: {outputs.shape}, Labels: {labels_batch.shape}. Skipping batch.")
                    print(f"  Original num bags in list: {len(list_of_patch_bags)}, Moved: {len(moved_list_of_patch_bags)}")
                    continue
            else:
                if feedback_type in [7, 8, 9, 10]: 
                    outputs = model(moved_list_of_patch_bags, model_org, attention_tool, moved_list_of_features) # Model takes the list of bags
                elif feedback_type in [11, 12, 13]: # ordinal
                    outputs, _, _, _ = model(moved_list_of_patch_bags) # Model takes the list of bags
                else:
                    outputs, _, _, _, _, _ = model(moved_list_of_patch_bags, model_org, attention_tool) # Model takes the list of bags

                # Ensure outputs and labels match in size after potential filtering
                if outputs.shape[0] != labels_batch.shape[0]:
                    print(f"Shape mismatch in {desc}! Outputs: {outputs.shape}, Labels: {labels_batch.shape}. Skipping batch.")
                    print(f"  Original num bags in list: {len(list_of_patch_bags)}, Moved: {len(moved_list_of_patch_bags)}")
                    continue

            # print("labels_batch:", labels_batch)
            # for k in moved_list_of_features:
            #     print("moved_list_of_features:", k)  # Debugging line to check the features being passed
            
            if feedback_type == 10:
                for i, labels in enumerate(labels_batch):
                    if moved_list_of_features[i][0] == 0 and moved_list_of_features[i][1] == 0 and moved_list_of_features[i][2] == -999:
                        # print("labels_batch org:", labels_batch[i])
                        labels_batch[i] = 0  # Set label to 0 if features are zero
            
            # print("labels_batch:", labels_batch)
            # for k in moved_list_of_features:
            #     print("moved_list_of_features:", k)  # Debugging line to check the features being passed
            targets = {}
            if feedback_type in [14]:
                kl_labels   = torch.tensor([l for l in labels_batch], device=labels_batch.device)
                jsnm_labels = torch.tensor([f[0] for f in moved_list_of_features], device=labels_batch.device)
                jsnl_labels = torch.tensor([f[1] for f in moved_list_of_features], device=labels_batch.device)
                
                osfm_labels = torch.tensor([f[2] for f in moved_list_of_features], device=labels_batch.device)
                ostm_labels = torch.tensor([f[3] for f in moved_list_of_features], device=labels_batch.device)
                ostl_labels = torch.tensor([f[4] for f in moved_list_of_features], device=labels_batch.device)
                osfl_labels = torch.tensor([f[5] for f in moved_list_of_features], device=labels_batch.device)

                # Mask: KL == 1 & OS == -999 → set OS to 0
                mask = (kl_labels == 1)
                osfm_labels = torch.where(mask & (osfm_labels == -999), torch.tensor(0, device=osfm_labels.device), osfm_labels)
                ostm_labels = torch.where(mask & (ostm_labels == -999), torch.tensor(0, device=ostm_labels.device), ostm_labels)
                ostl_labels = torch.where(mask & (ostl_labels == -999), torch.tensor(0, device=ostl_labels.device), ostl_labels)
                osfl_labels = torch.where(mask & (osfl_labels == -999), torch.tensor(0, device=osfl_labels.device), osfl_labels)


                targets.update({
                    "kl": kl_labels,
                    "jsnm": jsnm_labels,
                    "jsnl": jsnl_labels,
                    "osfm": osfm_labels,
                    "ostm": ostm_labels,
                    "ostl": ostl_labels,
                    "osfl": osfl_labels,                    
                })
                loss, loss_dict = criterion(outputs, targets)
            else:
                loss = criterion(outputs, labels_batch)

            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels_batch.size(0) # loss.item() is avg loss for batch
        num_processed_samples += labels_batch.size(0)


        if feedback_type in [14]:
            predicted = coral_multitask_predict(outputs) # dict of logits
            for task in oai_task_num_classes.keys():
                all_preds[task].extend(predicted[task].cpu().numpy())
                all_labels[task].extend(targets[task].cpu().numpy())
        else:
            if feedback_type in [11, 12, 13]:
                predicted = coral_predict(outputs) # logits
            else:
                _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

        if progress_bar:
            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / num_processed_samples if num_processed_samples > 0 else 0
    return avg_loss, all_labels, all_preds, num_processed_samples


# ---------------- Main Execution ---------------- #
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--training_type", type=str, default="original", choices=["original", "GradCAM", "GradCAMPlusPlus", "ScoreCAM", "AblationCAM", "LayerCAM"])
    parser.add_argument("--pre_ckpt", type=str, default="PRE_CHECKPOINT_DIR")
    parser.add_argument("--feedback_type", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], help="1: (w*cam*p + p), 2: (w(cam*p + p)), 3: (w*cam*p), 4: (w*cam*p + p), 5: (w*cam*p + w*p)")
    parser.add_argument("--note", type=str, default="", help="Additional note for the run name")

    num_features = 6
    
    data_part = "halfdata" if DATA_HALF else "wholedata"
    args = parser.parse_args()
    training_type = args.training_type
    PRE_CHECKPOINT_DIR = args.pre_ckpt
    feedback_type = args.feedback_type
    note = args.note
    CHECKPOINT_DIR = os.path.join("original_data", "V00", f"model_checkpoints_{NOW}_epoch200_finalckpt_100_feedback_{feedback_type}_{note}_feat_{num_features}")  # e.g., "att_lr1e-4_b16_0717_1245_feedback_1_onlyjsm_l"
    MEAN_STD_FILE_PATH = os.path.join(CHECKPOINT_DIR, "mean_std_train_patches.npy")

    print(f"Training type: {training_type}")
    print(f"Using device: {DEVICE}")
    print(f"Save Data to: {CHECKPOINT_DIR}")

    # Build a short but meaningful name
    timestamp = datetime.now().strftime('%m%d_%H%M')
    run_name = f"{training_type}_lr{LEARNING_RATE:.0e}_b{BATCH_SIZE}_{timestamp}_feedback_{feedback_type}_{data_part}_{note}_feat_{num_features}"  # e.g., "att_lr1e-4_b16_0717_1245"

    config = {
        "h5_file": H5_FILE,
        "mean_std_file_path": MEAN_STD_FILE_PATH,
        "pretrained_model_path": PRETRAINED_MODEL_PATH,
        "pre_ckpt_dir": PRE_CHECKPOINT_DIR,
        "save_path": CHECKPOINT_DIR,
        "num_classes": NUM_CLASSES,
        "feature_extractor_out_dim": FEATURE_EXTRACTOR_OUT_DIM,
        "aggregation_type": AGGREGATION_TYPE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "seed": SEED,
        "default_max_pixel_value": DEFAULT_MAX_PIXEL_VALUE,
        "training_type": training_type,
        "run_name": run_name,
    }
    print(config)
    
    # Make sure the directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # List of files to copy
    files_to_copy = ["my_train_feedback_all.py", "model.py", "dataset.py", "data_augmentation.py"]

    # Copy each file to the checkpoint directory
    if not DEBUG_MODE:
        for file in files_to_copy:
            if os.path.exists(file):
                shutil.copy(file, CHECKPOINT_DIR)
                print(f"Copied {file} to {CHECKPOINT_DIR}")
            else:
                print(f"WARNING: {file} not found and was not copied.")


    if WANDB:
        wandb.init(
            project="Knee_OA_MIL",
            name=run_name,
            config=config,
            tags=[
                training_type,
                AGGREGATION_TYPE,
                f"lr{LEARNING_RATE:.0e}",
                f"b{BATCH_SIZE}",
                f"s{SEED}"
            ],
            # group=f"{training_type}_{AGGREGATION_TYPE}",
        )

    # Write to JSON inside the checkpoint dir
    config_path = os.path.join(CHECKPOINT_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Config saved to: {config_path}")


    # 1. Load Patient IDs and filter valid samples
    with h5py.File(H5_FILE, 'r') as hf:
        base_ids = [pid.decode('utf-8') for pid in hf['patient_ids_order'][:]]
        groups, grades = [], []
        for pid in base_ids:
            for side in ["_L", "_R"]:
                group = pid + side
                if group in hf and hf[group]['kl_grade'][0] != -999 and hf[group]['patches'].shape[0] > 0:
                        groups.append(group)
                        grades.append(hf[group]['kl_grade'][0])

    print(f"Total valid samples: {len(groups)}")

    # 2. Train/Val/Test Split
    train_val, test, train_val_grades, _ = train_test_split(
        np.array(groups), np.array(grades), test_size=0.2, stratify=grades, random_state=SEED
    )
    train, val, _, _ = train_test_split(
        train_val, train_val_grades, test_size=0.25, stratify=train_val_grades, random_state=SEED
    )
    
    # if "9491446_R" in test:
    #    test.remove("9491446_R")
    
    # Convert back to lists if preferred by your Dataset class, or keep as numpy arrays
    train_pids = train.tolist()
    val_pids = val.tolist()
    test_pids = test.tolist()

    test_pids.remove("9491446_R") # bad image

    print(f"Total KNEE samples for training: {len(train_pids)}")
    print(f"Total KNEE samples for validation: {len(val_pids)}")
    print(f"Total KNEE samples for testing: {len(test_pids)}")
    
    

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # 3. Compute or Load Mean/Std
    if os.path.exists(MEAN_STD_FILE_PATH):
        mean, std = np.load(MEAN_STD_FILE_PATH)
    else:
        mean, std = calculate_mean_std(H5_FILE, train, MEAN_STD_FILE_PATH)

    print(f"Mean: {mean}, Std: {std}")

    # 4. Transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        CorrectBrightness(0.7, 1.3),
        CorrectContrast(0.7, 1.3),
        CorrectGamma(0.5, 2.5, res=8),
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist())
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),            # Convert numpy array to PIL
        transforms.ToTensor(),             # Convert PIL to tensor
        transforms.Normalize(mean.tolist(), std.tolist())
    ])

    # 5. Datasets and Loaders
    # Cut the train/val/test indices into halves
    if DATA_HALF:
        train_half = train[:len(train) // 2]
        val_half = val[:len(val) // 2]
        test_half = test[:len(test) // 2]
        print(f"KNEE samples for training: {len(train_half)}")
        print(f"KNEE samples for validation: {len(val_half)}")
        print(f"KNEE samples for testing: {len(test_half)}")
        train_ds = KneeMILDataset(H5_FILE, train_half.tolist(), transform=train_transform)
        val_ds = KneeMILDataset(H5_FILE, val_half.tolist(), transform=val_transform)
        test_ds = KneeMILDataset(H5_FILE, test_half, transform=val_transform)
    else:
        train_ds = KneeMILDataset(H5_FILE, train.tolist(), transform=train_transform)
        val_ds = KneeMILDataset(H5_FILE, val.tolist(), transform=val_transform)
        test_ds = KneeMILDataset(H5_FILE, test, transform=val_transform)

    train_loader = DataLoader(train_ds, BATCH_SIZE, True, collate_fn=mil_collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, BATCH_SIZE, False, collate_fn=mil_collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_ds, BATCH_SIZE, False, collate_fn=mil_collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)



    # 6. Model, Loss, Optimizer
    if feedback_type in [11, 12, 13]: # ordinal model
        model = CompleteMILOrdinalModel(FEATURE_EXTRACTOR_OUT_DIM, NUM_CLASSES, AGGREGATION_TYPE).to(DEVICE)
        model_org = None
    elif feedback_type in [14]: # multitask
        oai_task_num_classes={
            "kl": 5,   # 0–4 ordinal
            "jsnm": 4,  # 0–3 ordinal
            "jsnl": 4,  # 0–3 ordinal
        }
        model = CompleteMILOrdinal_MultiTask_Model(FEATURE_EXTRACTOR_OUT_DIM, NUM_CLASSES, oai_task_num_classes, AGGREGATION_TYPE).to(DEVICE)
        model_org = None
    else:
        model = CompleteMILCamModel_Attention_feedback(FEATURE_EXTRACTOR_OUT_DIM, NUM_CLASSES, training_type, feedback_type, AGGREGATION_TYPE, num_features=num_features).to(DEVICE)
        model_org = CompleteMILModel(FEATURE_EXTRACTOR_OUT_DIM, NUM_CLASSES, AGGREGATION_TYPE).to(DEVICE)
        model_org.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
    
    # <<<<<<< LOAD PRE-TRAINED WEIGHTS >>>>>>>
    # if os.path.exists(PRETRAINED_MODEL_PATH):
    #     try:
    #         model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
    #         print(f"Successfully loaded pre-trained weights from: {PRETRAINED_MODEL_PATH}")
    #     except Exception as e:
    #         print(f"Error loading pre-trained weights: {e}. Training from scratch.")
    # else:
    #     print(f"Pre-trained model path not found: {PRETRAINED_MODEL_PATH}. Training from scratch.")

    # weighted loss for class imbalance
    train_kl_grades = []
    with h5py.File(H5_FILE, 'r') as hf:
        for group_name in train_ds.sample_group_names: # Assuming train_dataset is initialized
            train_kl_grades.append(hf[group_name]['kl_grade'][0])

    class_counts = np.bincount(train_kl_grades, minlength=NUM_CLASSES)
    # Avoid division by zero if a class is missing in training (should ideally not happen with good splits)
    # class_weights_raw = 1.0 / (class_counts + 1e-6) # Add epsilon for stability
    # class_weights_raw[1] *= 2.0 # Double the weight for class 1
    # class_weights_normalized = class_weights_raw / np.sum(class_weights_raw) * NUM_CLASSES # Optional normalization
    # class_weights_tensor = torch.tensor(class_weights_normalized, dtype=torch.float).to(DEVICE)

    print(f"Class counts in training set: {class_counts}")
    
    if feedback_type in [13]: # class weights
        class_weights = compute_effective_class_weights(class_counts, num_classes=NUM_CLASSES, beta=0.9999)
        class_weights = np.array(class_weights)
        class_weights[1, :] = [class_weights[1,0]*2.0, class_weights[1,1]*2.0] # Double the weight for class 1
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    else: # sample weights
        class_weights_raw = 1.0 / (class_counts + 1e-6) # Add epsilon for stability
        # class_weights_raw[1] *= 2.0 # Double the weight for class 1
        class_weights_normalized = class_weights_raw / np.sum(class_weights_raw) * NUM_CLASSES # Optional normalization
        class_weights_tensor = torch.tensor(class_weights_normalized, dtype=torch.float).to(DEVICE)

    print(f"Using class weights: {class_weights_tensor}")
    if feedback_type == 11:
        criterion = CoralLossWeighted(class_weights=class_weights_tensor)
    elif feedback_type == 12:
        criterion = CoralFocalLoss(class_weights=class_weights_tensor, gamma=2.0, alpha=0.25)

    elif feedback_type == 13:
        criterion = CoralLossEffective(threshold_weights=class_weights_tensor)
    elif feedback_type == 14:
        criterion = MultiTask_CoralFocalLoss(oai_task_num_classes, is_learn_task_weights=True, class_weights=class_weights_tensor)

    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5) # For val_loss

    best_val_accuracy = 0.0
    best_val_loss = np.inf
    best_val_kappa = -1.0
    best_val_f1 = 0.0

    best_val_accuracy_kl = 0.0
    best_val_kappa_kl = -1.0
    best_val_f1_kl = 0.0
    best_model_path_fixed = os.path.join(CHECKPOINT_DIR, "best_model.pth") # Fixed name for best model


    for epoch in range(NUM_EPOCHS):
        torch.cuda.empty_cache()
        epoch_num = epoch + 1

        current_lr = optimizer.param_groups[0]['lr']
        # writer.add_scalar('Hyperparameters/LearningRate', current_lr, epoch_num) # Log it

        # Training phase
        train_loss, train_labels, train_preds, processed_train_samples = run_epoch(
            train_loader, model, model_org, criterion, optimizer, DEVICE, is_training=True, training_type=training_type, feedback_type=feedback_type, oai_task_num_classes=oai_task_num_classes,
            desc=f"Epoch {epoch_num}/{NUM_EPOCHS} [Train]"
        )
        if processed_train_samples > 0:
            if feedback_type in [14]:
                task_accuracy = {}
                task_f1 = {}
                task_kappa = {}
                for task in train_labels.keys():
                    labels = train_labels[task]
                    preds = train_preds[task]
                    acc = accuracy_score(labels, preds)
                    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
                    kappa = cohen_kappa_score(labels, preds, weights="quadratic")
                    task_accuracy[task] = acc
                    task_f1[task] = f1
                    task_kappa[task] = kappa
            
                    if WANDB:
                        wandb.log({
                            f"train/{task}_accuracy": acc,
                            f"train/{task}_f1_weighted": f1,
                            f"train/{task}_kappa": kappa,
                            "epoch": epoch_num
                        })
                    print(f"[Train] {task} - Acc: {acc:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}")

                # aggregate across tasks (mean of metrics)
                train_accuracy = np.mean(list(task_accuracy.values()))
                train_f1 = np.mean(list(task_f1.values()))
                train_kappa = np.mean(list(task_kappa.values()))
            else:
                train_accuracy = accuracy_score(train_labels, train_preds)
                train_f1 = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
                train_kappa = cohen_kappa_score(train_labels, train_preds, weights="quadratic") # Added Kappa
            
            if WANDB:
                wandb.log({
                    "train/loss": train_loss,
                    "train/accuracy": train_accuracy,
                    "train/f1_weighted": train_f1,
                    "train/kappa": train_kappa,
                    "learning_rate": current_lr,
                    "epoch": epoch_num
                })
                print(f"Epoch {epoch_num} Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train F1: {train_f1:.4f}, Train Kappa: {train_kappa:.4f}")
        else:
            print(f"Epoch {epoch_num} - No samples processed during training.")

        # Validation phase
        # No optimizer needed for validation
        val_loss, val_labels, val_preds, processed_val_samples = run_epoch(
            val_loader, model, model_org, criterion, None, DEVICE, is_training=False, training_type=training_type, feedback_type=feedback_type, oai_task_num_classes=oai_task_num_classes,
            desc=f"Epoch {epoch_num}/{NUM_EPOCHS} [Val]"
        )

        scheduler.step(val_loss)

        if processed_val_samples > 0:
            if feedback_type in [14]:
                task_accuracy = {}
                task_f1 = {}
                task_kappa = {}
                for task in val_labels.keys():
                    labels = val_labels[task]
                    preds = val_preds[task]
                    acc = accuracy_score(labels, preds)
                    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
                    kappa = cohen_kappa_score(labels, preds, weights="quadratic")
                    task_accuracy[task] = acc
                    task_f1[task] = f1
                    task_kappa[task] = kappa
            
                    if WANDB:
                        wandb.log({
                            f"val/{task}_accuracy": acc,
                            f"val/{task}_f1_weighted": f1,
                            f"val/{task}_kappa": kappa,
                            "epoch": epoch_num
                        })
                    print(f"[Val] {task} - Acc: {acc:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}")

                # aggregate across tasks (mean of metrics)
                val_accuracy = np.mean(list(task_accuracy.values()))
                val_f1 = np.mean(list(task_f1.values()))
                val_kappa  = np.mean(list(task_kappa.values()))
            else:
                val_accuracy = accuracy_score(val_labels, val_preds)
                val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
                val_kappa = cohen_kappa_score(val_labels, val_preds, weights="quadratic") # Added Kappa
            
            if WANDB:
                wandb.log({
                    "val/loss": val_loss,
                    "val/accuracy": val_accuracy,
                    "val/f1_weighted": val_f1,
                    "val/kappa": val_kappa,
                    "learning_rate": current_lr,
                    "epoch": epoch_num
                })
            print(f"Epoch {epoch_num} Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val Kappa: {val_kappa:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(),
                        os.path.join(CHECKPOINT_DIR, f"best_model_{training_type}_val_acc.pth"))
                print(f"  Saved new best acc model ({training_type}) (Val Acc: {val_accuracy:.4f})")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(),
                        os.path.join(CHECKPOINT_DIR, f"best_model_{training_type}_val_f1.pth"))
                print(f"  Saved new best f1 model ({training_type}) (Val F1: {val_f1:.4f})")

            if val_kappa > best_val_kappa:
                best_val_kappa = val_kappa
                torch.save(model.state_dict(),
                        os.path.join(CHECKPOINT_DIR, f"best_model_{training_type}_val_kappa.pth"))
                print(f"  Saved new best kappa model ({training_type}) (Val kappa: {val_kappa:.4f})")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(),
                        os.path.join(CHECKPOINT_DIR, f"best_model_{training_type}_val_loss.pth"))
                print(f"  Saved new best loss model ({training_type}) (Val loss: {val_loss:.4f})")

            if "kl" in task_accuracy:
                if task_accuracy["kl"] > best_val_accuracy_kl:
                    best_val_accuracy_kl = task_accuracy["kl"]
                    torch.save(model.state_dict(),
                            os.path.join(CHECKPOINT_DIR, f"best_model_{training_type}_kl_val_acc.pth"))
                    print(f"  Saved new best KL acc model ({training_type}) (KL Acc: {task_accuracy['kl']:.4f})")

                if task_kappa["kl"] > best_val_kappa_kl:
                    best_val_kappa_kl = task_kappa["kl"]
                    torch.save(model.state_dict(),
                            os.path.join(CHECKPOINT_DIR, f"best_model_{training_type}_kl_val_kappa.pth"))
                    print(f"  Saved new best KL kappa model ({training_type}) (KL Kappa: {task_kappa['kl']:.4f})")
                
                if task_f1["kl"] > best_val_f1_kl:
                    best_val_f1_kl = task_f1["kl"]
                    torch.save(model.state_dict(),
                            os.path.join(CHECKPOINT_DIR, f"best_model_{training_type}_kl_val_f1.pth"))
                    print(f"  Saved new best KL F1 model ({training_type}) (KL F1: {task_f1['kl']:.4f})")
        else:
            print(f"Epoch {epoch_num} - No samples processed during validation.")
        print("-" * 60)

    print("Training finished.")
    if WANDB:
        wandb.finish()