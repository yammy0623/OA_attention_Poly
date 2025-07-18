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
from model import CompleteMILModel, CompleteMILCamModel
import argparse
import matplotlib.pyplot as plt
import wandb
import json
from datetime import datetime
import shutil

# Get today’s date and time in YYYYMMDD_HHMM format
now = datetime.now().strftime('%Y%m%d_%H%M')

# ---------------- Configuration ---------------- #
H5_FILE = rf"model_checkpoints_tnc_final\knee_patches_patient_grouped_16_100.h5"
# CHECKPOINT_DIR = rf"original_data\V00\model_checkpoints_0710_epoch200"
# PRE_CHECKPOINT_DIR = rf"original_data\V00\model_checkpoints"
PRE_CHECKPOINT_DIR = rf"model_checkpoints_tnc_final"
# Build your checkpoint directory string
CHECKPOINT_DIR = rf"original_data\V00\model_checkpoints_{now}_epoch200_finalckpt_100"
MEAN_STD_FILE_PATH = os.path.join(CHECKPOINT_DIR, "mean_std_train_patches.npy")
PRETRAINED_MODEL_PATH = os.path.join(PRE_CHECKPOINT_DIR, "best_model_val_acc.pth")

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


def run_epoch(loader, model, model_org, criterion, optimizer, device, is_training, training_type, desc=""):
    model.train() if is_training else model.eval()
    model_org.eval()

    total_loss, all_preds, all_labels, num_processed_samples = 0.0, [], [], 0

    progress_bar = tqdm(loader, desc=desc, leave=False)

    for list_of_patch_bags, labels_batch in progress_bar:
        if list_of_patch_bags is None or not list_of_patch_bags:
            # This might happen if collate_fn decides a batch is entirely invalid
            print(f"Warning: Skipped an entirely invalid batch in {desc}.")
            continue

        moved_list_of_patch_bags = []
        valid_indices_in_batch = [] # Keep track of which original labels correspond to moved bags

        for i, bag in enumerate(list_of_patch_bags):
            if bag.nelement() > 0:  # Check if bag is not empty
                moved_list_of_patch_bags.append(bag.to(DEVICE, non_blocking=PIN_MEMORY))
                valid_indices_in_batch.append(i)
        
        # moved_list_of_patch_bags = torch.stack(moved_list_of_patch_bags).to(DEVICE)

        if not moved_list_of_patch_bags: # If all bags in this batch were empty after filtering
            # print(f"Warning: All bags in current batch for {epoch_desc} were empty. Skipping.")
            continue

        # Select labels corresponding to the valid bags
        labels_batch = labels_batch[valid_indices_in_batch].to(DEVICE, non_blocking=PIN_MEMORY)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training): # Context manager for gradients
            outputs, _ = model(moved_list_of_patch_bags, model_org, training_type) # Model takes the list of bags

            # Ensure outputs and labels match in size after potential filtering
            if outputs.shape[0] != labels_batch.shape[0]:
                print(f"Shape mismatch in {desc}! Outputs: {outputs.shape}, Labels: {labels_batch.shape}. Skipping batch.")
                print(f"  Original num bags in list: {len(list_of_patch_bags)}, Moved: {len(moved_list_of_patch_bags)}")
                continue

            loss = criterion(outputs, labels_batch)

            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels_batch.size(0) # loss.item() is avg loss for batch
        num_processed_samples += labels_batch.size(0)

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
    args = parser.parse_args()
    training_type = args.training_type

    print(f"Training type: {training_type}")
    print(f"Using device: {DEVICE}")
    print(f"Save Data to: {CHECKPOINT_DIR}")

    # Build a short but meaningful name
    timestamp = datetime.now().strftime('%m%d_%H%M')
    run_name = f"{training_type}_lr{LEARNING_RATE:.0e}_b{BATCH_SIZE}_{timestamp}"  # e.g., "att_lr1e-4_b16_0717_1245"

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
    files_to_copy = ["my_train.py", "model.py", "my_inference.py", "dataset.py", "data_augmentation.py"]

    # Copy each file to the checkpoint directory
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy(file, CHECKPOINT_DIR)
            print(f"Copied {file} to {CHECKPOINT_DIR}")
        else:
            print(f"WARNING: {file} not found and was not copied.")


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

    # test_pids.remove("9491446_R") # bad image

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
        transforms.Normalize(mean.tolist(), std.tolist())
    ])

    # 5. Datasets and Loaders
    train_ds = KneeMILDataset(H5_FILE, train.tolist(), transform=train_transform)
    val_ds = KneeMILDataset(H5_FILE, val.tolist(), transform=val_transform)
    test_ds = KneeMILDataset(H5_FILE, test, transform=val_transform)

    train_loader = DataLoader(train_ds, BATCH_SIZE, True, collate_fn=mil_collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, BATCH_SIZE, False, collate_fn=mil_collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_ds, BATCH_SIZE, False, collate_fn=mil_collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)



    # 6. Model, Loss, Optimizer
    model = CompleteMILCamModel(FEATURE_EXTRACTOR_OUT_DIM, NUM_CLASSES, training_type, AGGREGATION_TYPE).to(DEVICE)
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
    class_weights_raw = 1.0 / (class_counts + 1e-6) # Add epsilon for stability
    class_weights_normalized = class_weights_raw / np.sum(class_weights_raw) * NUM_CLASSES # Optional normalization
    class_weights_tensor = torch.tensor(class_weights_normalized, dtype=torch.float).to(DEVICE)
    print(f"Using class weights: {class_weights_tensor}")
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5) # For val_loss

    best_val_accuracy = 0.0
    best_val_loss = np.inf
    best_val_kappa = -1.0

    best_model_path_fixed = os.path.join(CHECKPOINT_DIR, "best_model.pth") # Fixed name for best model


    for epoch in range(NUM_EPOCHS):
        torch.cuda.empty_cache()
        epoch_num = epoch + 1

        current_lr = optimizer.param_groups[0]['lr']
        # writer.add_scalar('Hyperparameters/LearningRate', current_lr, epoch_num) # Log it

        # Training phase
        train_loss, train_labels, train_preds, processed_train_samples = run_epoch(
            train_loader, model, model_org, criterion, optimizer, DEVICE, is_training=True, training_type=training_type,
            desc=f"Epoch {epoch_num}/{NUM_EPOCHS} [Train]"
        )
        if processed_train_samples > 0:
            train_accuracy = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
            train_kappa = cohen_kappa_score(train_labels, train_preds, weights="quadratic") # Added Kappa
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
        val_loss, val_labels, val_preds, processed_val_samples = run_epoch(
            val_loader, model, model_org, criterion, None, DEVICE, is_training=False, training_type=training_type,# No optimizer needed for validation
            desc=f"Epoch {epoch_num}/{NUM_EPOCHS} [Val]"
        )

        scheduler.step(val_loss)

        if processed_val_samples > 0:
            val_accuracy = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
            val_kappa = cohen_kappa_score(val_labels, val_preds, weights="quadratic") # Added Kappa
            wandb.log({
                "val/loss": val_loss,
                "val/accuracy": val_accuracy,
                "val/f1_weighted": val_f1,
                "train/kappa": val_kappa,
                "learning_rate": current_lr,
                "epoch": epoch_num
            })
            # writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch_num)
            # writer.add_scalars('Accuracy', {'train': train_accuracy, 'val': val_accuracy}, epoch_num)
            # writer.add_scalars('F1_score_weighted', {'train': train_f1, 'val': val_f1}, epoch_num)
            # writer.add_scalars('Kappa_weighted', {'train': train_kappa, 'val': val_kappa}, epoch_num)
            print(f"Epoch {epoch_num} Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val Kappa: {val_kappa:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(),
                    os.path.join(CHECKPOINT_DIR, f"best_model_{training_type}_val_acc.pth"))
            print(f"  Saved new best acc model ({training_type}) (Val Acc: {val_accuracy:.4f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                    os.path.join(CHECKPOINT_DIR, f"best_model_{training_type}_val_loss.pth"))
            print(f"  Saved new best loss model ({training_type}) (Val loss: {val_loss:.4f})")

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            torch.save(model.state_dict(),
                    os.path.join(CHECKPOINT_DIR, f"best_model_{training_type}_val_kappa.pth"))
            print(f"  Saved new best kappa model ({training_type}) (Val kappa: {val_kappa:.4f})")
        else:
            print(f"Epoch {epoch_num} - No samples processed during validation.")
        print("-" * 60)

    print("Training finished.")
    wandb.finish()