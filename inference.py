import os
import shutil
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from config import build_config
from dataset import KneeMILDataset, mil_collate_fn
from losses import coral_predict, coral_multitask_predict
from myutils import (
    calculate_mean_std,
    build_CAM_attention_tool,
    compute_metrics,
    get_criterion,
    labels_to_levels,
    create_transforms,
    prepare_data,
    get_model,
    get_model_org
)

class Config:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)


def run_epoch(loader, model, model_org, criterion, optimizer, device, is_training, config, desc=""):
    """
    Run one epoch of training/validation
    config: parsed argparse with flags like config.use_multitask, config.use_ordinal, config.training_type
    """
    model.train() if is_training else model.eval()
    total_loss, num_processed_samples = 0.0, 0

    # Prepare prediction containers
    if config.multitask_type == "off":
        all_preds, all_labels, all_probs = [], [], []
    else:
        all_preds = {task: [] for task in config.OARSI_TASKS.keys()}
        all_labels = {task: [] for task in config.OARSI_TASKS.keys()}
        all_probs = {task: [] for task in config.OARSI_TASKS.keys()}
        

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
                
                if config.lossfcn_type == "CoralLoss_MultiTask":
                    targets_levels = {}
                    for k, v in targets.items():
                        num_classes = config.OARSI_TASKS[k]  # your dict of num classes per task
                        targets_levels[k] = labels_to_levels(v, num_classes)
                    loss, loss_dict = criterion(outputs, targets_levels)
                else:
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
                predicted, probs = coral_predict(outputs)
            elif config.predict_criteria == "Max":
                _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        else:
            if config.predict_criteria == "Coral_Multitask":
                predicted, probs = coral_multitask_predict(outputs)

            for task in config.OARSI_TASKS.keys():
                all_preds[task].extend(predicted[task][0].cpu().numpy())
                all_labels[task].extend(targets[task].cpu().numpy())
                all_probs[task].extend(probs[task][0].cpu().numpy())

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / num_processed_samples if num_processed_samples > 0 else 0
    return avg_loss, all_labels, all_preds, all_probs

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

    test_loader = DataLoader(test_ds, config.BATCH_SIZE, False, collate_fn=mil_collate_fn,
                             num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

    # ----------------- Model ----------------- #
    model = get_model(config)
    model_org = get_model_org(config)

    model.load_state_dict(torch.load(
            os.path.join(config.CHECKPOINT_DIR, f"best_model_{config.inference_target}_acc.pth"), 
            map_location=config.DEVICE
        ))
    if model_org:
        model_org.load_state_dict(torch.load(config.PRETRAINED_MODEL_PATH, map_location=config.DEVICE))
        
    # ----------------- Loss & Optimizer ----------------- #
    class_weights_tensor = None
    criterion = get_criterion(config.lossfcn_type, class_weights_tensor, config.OARSI_TASKS)
    optimizer = None

    # ----------------- Training Loop ----------------- #

    # Train
    test_loss, test_labels, test_preds, test_probs = run_epoch(
        test_loader, model, model_org, criterion, optimizer, config.DEVICE,
        is_training=False, config=config,
        desc=f"[Testing]"
    )
    print(f"\nTest Loss: {test_loss:.4f}")
    results = [f"Test Loss: {test_loss:.4f}"]
    test_metrics = compute_metrics(config.multitask_type, test_labels, test_preds)

        
    # for task in test_labels.keys():
    task = config.inference_target
    labels = test_labels[task]
    preds = test_preds[task]

    acc = test_metrics[task]["acc"]
    f1 = test_metrics[task]["f1"]
    kappa = test_metrics[task]["kappa"]
    print(f"[Test] {task} - Acc: {acc:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}")
    results.append(f"[Test] {task} - Acc: {acc:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}")

    num_classes = config.OARSI_TASKS[task]
    target_names = [f"{task.upper()} {i}" for i in range(num_classes)]

    report = classification_report(labels, preds, target_names=target_names, zero_division=0)
    print(report)
    results.append(f"\n[{task}]\n" + report)
    
    ConfusionMatrixDisplay.from_predictions(
        labels, preds, normalize="true", cmap=plt.cm.Greens, values_format='.2f'
    )
    plt.savefig(os.path.join(config.CHECKPOINT_DIR, f"cm_{task}.eps"), format='eps')
    plt.savefig(os.path.join(config.CHECKPOINT_DIR, f"cm_{task}.png"), format='png')
    plt.close()
    save_path = os.path.join(config.CHECKPOINT_DIR, "inference_result.txt")
    with open(save_path, "w") as f:
        for line in results:
            f.write(line + "\n")

    print("Inference finished.")


if __name__ == "__main__":
    from config import build_config
    config_dict = build_config()
    cfg = Config(config_dict)
    cfg.DEBUG_MODE = True
    cfg.WANDB = False
    main(cfg)
    