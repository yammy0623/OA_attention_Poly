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
from model import CompleteMILModel, CompleteMILCamModel, CompleteMILCamModel_Attention_feedback, CompleteMILOrdinalModel
import matplotlib.pyplot as plt
import argparse
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, LayerCAM
import shutil

# ---------------- Configuration ---------------- #
# H5_FILE = os.path.join("model_checkpoints_tnc_final", "knee_patches_patient_grouped_16_100.h5")
H5_FILE = "knee_patches_patient_grouped_16_100_all_feature.h5"
PRE_CHECKPOINT_DIR = "model_checkpoints_tnc_final"
CHECKPOINT_DIR = os.path.join("original_data", "V00", f"model_checkpoints_20250718_1217_epoch200_finalckpt_100")
MEAN_STD_FILE_PATH = os.path.join(CHECKPOINT_DIR, "mean_std_train_patches.npy")
PRETRAINED_MODEL_PATH = os.path.join(PRE_CHECKPOINT_DIR, "best_model_val_acc.pth")

NUM_CLASSES = 5
FEATURE_EXTRACTOR_OUT_DIM = 128
AGGREGATION_TYPE = 'attention'
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 100
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

def coral_predict(logits):
    """
    logits: (batch_size, K-1)
    return: predicted class (batch_size,)
    """
    prob = torch.sigmoid(logits)   # (batch, K-1)
    # 檢查從左到右哪個 cutpoint 變成 <0.5
    preds = torch.sum(prob > 0.5, dim=1)
    return preds

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

def plot_patches_grid_with_heatmaps(
    patches_list,
    heatmaps_list,
    patch_indices_map,
    num_cols=8,
    figure_title="",
    patch_cmap='gray',
    heatmap_cmap=plt.get_cmap('Reds'),
    base_heatmap_alpha=0.3,
    attention_scores_norm=None, # Optional: for alpha modulation
    title_fontsize=8,
    interpolation_method='nearest',
    training_type="original",
    method_name="GradCAM"
):
    """
    Displays a grid of image patches with overlaid heatmaps.

    Args:
        patches_list (list/np.array): List of image patches (each a NumPy array).
        heatmaps_list (list/np.array): List of heatmaps corresponding to patches.
        patch_indices_map (list/np.array): List of original indices/labels for titles.
        num_cols (int): Number of columns in the grid.
        figure_title (str): Overall title for the figure.
        patch_cmap (str): Colormap for the base patches.
        heatmap_cmap (str): Colormap for the heatmaps.
        base_heatmap_alpha (float): Base alpha transparency for heatmaps.
        attention_scores_norm (list/np.array, optional): Normalized attention scores.
                                                        If provided, heatmap_alpha = base_heatmap_alpha * score.
        title_fontsize (int): Font size for individual patch titles.
        interpolation_method (str): Interpolation method for imshow.
    """
    import math

    if not patches_list:
        print("No patches to display.")
        return

    n_patches = len(patches_list)
    if n_patches != len(heatmaps_list) or n_patches != len(patch_indices_map):
        print("Error: Mismatch in lengths of patches, heatmaps, or patch_indices_map.")
        return

    n_rows = math.ceil(n_patches / num_cols)

    fig, axes = plt.subplots(n_rows, num_cols, figsize=(num_cols * 2, n_rows * 2.2)) # Slightly taller for titles
    
    # Flatten axes array for easier indexing, handling single row/col cases
    if n_rows * num_cols > 1:
        axes = axes.flatten()
    elif n_rows * num_cols == 1: # Single subplot
        axes = [axes] 
    else: # No subplots if n_patches is 0 (already handled)
        return

    for idx in range(n_patches):
        ax = axes[idx]
        patch = patches_list[idx]
        heatmap = heatmaps_list[idx]

        # Prepare patch (squeeze to 2D if needed)
        patch_np = patch.squeeze()

        # Plot base patch
        ax.imshow(patch_np, cmap=patch_cmap, interpolation=interpolation_method)

        # Determine heatmap alpha
        current_alpha = base_heatmap_alpha
        if attention_scores_norm is not None and idx < len(attention_scores_norm):
            current_alpha *= attention_scores_norm[idx]
            current_alpha = np.clip(current_alpha, 0.0, 1.0) # Ensure alpha is valid

        # Overlay heatmap
        ax.imshow(heatmap, cmap=heatmap_cmap, alpha=current_alpha, interpolation=interpolation_method)

        # Set title and turn off axis
        ax.set_title(f"Point {patch_indices_map[idx]}", fontsize=title_fontsize)
        ax.axis('off')

    # Hide any unused subplots
    for idx in range(n_patches, n_rows * num_cols):
        axes[idx].axis('off')

    if figure_title:
        fig.suptitle(figure_title, fontsize=title_fontsize + 4, y=0.99) # Adjust y to prevent overlap

    plt.tight_layout(rect=[0, 0, 1, 0.95 if figure_title else 0.98]) # Adjust rect for suptitle
    # plt.show()
    plt.savefig(os.path.join(IMG_SAVE_PATH, f"heatmap_{training_type}_vis_{method_name}.eps"), format='eps')
    plt.savefig(os.path.join(IMG_SAVE_PATH, f"heatmap_{training_type}_vis_{method_name}.png"), format='png')

def process_CAM(model, target_layer, target_class, patch_bag_tensor, patches_test, training_type):
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, LayerCAM
    cam = GradCAM(
        model=model.patch_feature_extractor,     
        target_layers=target_layer,
        )
    campp = GradCAMPlusPlus(
        model=model.patch_feature_extractor,     
        target_layers=target_layer,
        )
    scam = ScoreCAM(
        model=model.patch_feature_extractor,     
        target_layers=target_layer,
        )
    acam = AblationCAM(
        model=model.patch_feature_extractor,    
        target_layers=target_layer,
        )
    lcam = LayerCAM(
        model=model.patch_feature_extractor,     
        target_layers=target_layer,
        )
    # batch_size = 41
    batch_size = patch_bag_tensor.shape[0]
    # 1) 準備 targets list，同一個 target_class 重複 batch_size 次
    targets = [ ClassifierOutputTarget(target_class) ] * batch_size

    # 2) 一次跑 CAM，回傳 shape = (batch_size, H, W)
    grayscale_cams = cam(
        input_tensor=patch_bag_tensor,   # shape [41,1,16,16]
        targets=targets
    )
    grayscale_camspp = campp(
        input_tensor=patch_bag_tensor,   # shape [41,1,16,16]
        targets=targets
    )
    grayscale_scam = scam(
        input_tensor=patch_bag_tensor,   # shape [41,1,16,16]
        targets=targets
    )
    grayscale_acam = acam(
        input_tensor=patch_bag_tensor,   # shape [41,1,16,16]
        targets=targets
    )
    grayscale_lcam = lcam(
        input_tensor=patch_bag_tensor,   # shape [41,1,16,16]
        targets=targets
    )
    cam_data_sources = {
        "GradCAM": grayscale_cams,
        "GradCAM++": grayscale_camspp,
        "ScoreCAM": grayscale_scam,
        "AblationCAM": grayscale_acam,
        "LayerCAM": grayscale_lcam
    }
    # Calculate PATCH_POINT_INDICES (do this once)
    range1 = np.arange(9, 27)  # Indices 9 to 26
    range2 = np.arange(44, 67) # Indices 44 to 66
    PATCH_POINT_INDICES = np.concatenate([range1, range2])
    grayscale_ensemble = None
    for item in cam_data_sources:
        if grayscale_ensemble is None:
            grayscale_ensemble = 1 / len(cam_data_sources) * cam_data_sources[item]
        else:
            grayscale_ensemble = grayscale_ensemble + 1 / len(cam_data_sources) * cam_data_sources[item]

    cam_data_sources["ensemble"] = grayscale_ensemble

    for method_name, heatmaps in cam_data_sources.items():
        print(f"Displaying grid for: {method_name}")
        # Make sure 'patches_test' and 'PATCH_POINT_INDICES' are defined and have the correct data
        # Also, ensure 'heatmaps' (i.e., grayscale_cams, etc.) are defined.
        # If 'att_scores_norm' is used, ensure it's defined too.
        
        # Example:
        # if 'patches_test' not in globals() or 'PATCH_POINT_INDICES' not in globals() or method_name not in globals():
        #    print(f"Skipping {method_name} due to missing data. Please define patches_test, PATCH_POINT_INDICES, and heatmaps.")
        #    continue

        plot_patches_grid_with_heatmaps(
            patches_list=patches_test,             # Replace with your actual patches data
            heatmaps_list=heatmaps,                # This will be grayscale_cams, grayscale_camsp, etc.
            patch_indices_map=PATCH_POINT_INDICES, # Replace with your actual indices
            figure_title=f"{method_name} Heatmaps on Patches",
            # attention_scores_norm=att_scores_norm, # Uncomment if using this
            base_heatmap_alpha=0.3, # Explicitly setting the alpha from your original code
            training_type=training_type, 
            method_name=method_name
        )


def run_epoch(loader, model, model_org, criterion, optimizer, device, is_training, training_type, feedback_type, desc=""):
    model.eval()
    total_loss, all_preds, all_labels, num_samples = 0.0, [], [], 0
    gradcam_type = training_type
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

    for bags, labels, group_name, list_of_features in tqdm(loader, desc=desc, leave=False):
        if not bags:
            continue

        moved_bags = []
        valid_indices = []
        moved_list_of_features = []

        for i, bag in enumerate(bags):
            if bag.nelement() > 0:
                moved_bags.append(bag.to(device))
                valid_indices.append(i)
        for i in valid_indices:
                # feat = list_of_features[i][0][0].unsqueeze(0)
                # feat = list_of_features[i][0][:2]
                feat = list_of_features[i][0]
                moved_list_of_features.append(feat.to(DEVICE, non_blocking=PIN_MEMORY))

        if not moved_bags:
            continue

        labels = labels[valid_indices].to(device)

        if is_training:
            optimizer.zero_grad()

        # print("attention_tool", attention_tool)
        with torch.no_grad():
            if feedback_type in [7, 8, 9, 10]:
                outputs = model(moved_bags, model_org, attention_tool, moved_list_of_features) # Model takes the list of bags
            elif feedback_type in [11]:
                outputs, _, _, _ = model(moved_bags)
            else:
                outputs, _, _, _, _, _ = model(moved_bags, model_org, attention_tool)
            if outputs.size(0) != labels.size(0):
                print(f"Skipping batch due to shape mismatch: {outputs.shape} vs {labels.shape}")
                continue

            if feedback_type == 10:
                for i, _ in enumerate(labels):
                    if moved_list_of_features[i][0] == 0 and moved_list_of_features[i][1] == 0 and moved_list_of_features[i][2] == -999:
                        # print("labels_batch org:", labels_batch[i])
                        labels[i] = 0  # Set label to 0 if features are zero

            loss = criterion(outputs, labels)
            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        
        if feedback_type == 11:
            predicted = coral_predict(outputs) # logits
        else:
            _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        num_samples += labels.size(0)

        # false_neg_1_to_0_ids = [id_ for id_, a, p in zip(group_name, labels.cpu().numpy(),  predicted.cpu().numpy()) if a == 1 and p == 0]

        # print("False negatives for class 1 predicted as 0:", false_neg_1_to_0_ids)
       
        # print("ids", ids)
        # print("actual labels:", labels.cpu().numpy())
        # print("predicted labels:", predicted.cpu().numpy())
        # a


    avg_loss = total_loss / num_samples if num_samples else 0
    return avg_loss, all_labels, all_preds


# ---------------- Main Execution ---------------- #
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--training_type", type=str, default="original", choices=["original", "GradCAM", "GradCAMPlusPlus", "ScoreCAM", "AblationCAM", "LayerCAM"])
    parser.add_argument("--pre_ckpt", type=str, default="PRE_CHECKPOINT_DIR")
    parser.add_argument("--current_ckpt", type=str, default="CHECKPOINT_DIR")
    parser.add_argument("--feedback_type", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], help="1: (w*cam*p + p), 2: (w(cam*p + p)), 3: (w*cam*p), 4: (w*cam*p + p), 5: (w*cam*p + w*p)")


    
    args = parser.parse_args()
    training_type = args.training_type
    PRE_CHECKPOINT_DIR = args.pre_ckpt
    CHECKPOINT_DIR = args.current_ckpt
    CHECKPOINT_DIR = os.path.join("original_data", "V00", CHECKPOINT_DIR)
    feedback_type = args.feedback_type
    IMG_SAVE_PATH=CHECKPOINT_DIR

    # List of files to copy
    files_to_copy = ["my_inference_feedback_all.py"]

    # Copy each file to the checkpoint directory
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy(file, CHECKPOINT_DIR)
            print(f"Copied {file} to {CHECKPOINT_DIR}")
        else:
            print(f"WARNING: {file} not found and was not copied.")
    print(f"Using device: {DEVICE}")

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
    if feedback_type == 11:
        model = CompleteMILOrdinalModel(FEATURE_EXTRACTOR_OUT_DIM, NUM_CLASSES, AGGREGATION_TYPE).to(DEVICE)
        model_org = None
    else:
        model = CompleteMILCamModel_Attention_feedback(FEATURE_EXTRACTOR_OUT_DIM, NUM_CLASSES, training_type, feedback_type, AGGREGATION_TYPE, num_features=6).to(DEVICE)
        model_org = CompleteMILModel(FEATURE_EXTRACTOR_OUT_DIM, NUM_CLASSES, AGGREGATION_TYPE).to(DEVICE)
    
    with h5py.File(H5_FILE, 'r') as hf:
        train_grades = [hf[group]['kl_grade'][0] for group in train_ds.sample_group_names]
    class_counts = np.bincount(train_grades, minlength=NUM_CLASSES)
    weights = 1.0 / (class_counts + 1e-6)
    weights /= weights.sum()
    class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    print(f"Class Weights: {class_weights}")

    if feedback_type == 11:
        criterion = CoralLossWeighted(class_weights=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    # 7. Test Inference
    if os.path.exists(CHECKPOINT_DIR):
        model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, f"best_model_{training_type}_val_acc.pth"), map_location=DEVICE))
        model_org.load_state_dict(torch.load(PRETRAINED_MODEL_PATH , map_location=DEVICE))
        
        test_loss, test_labels, test_preds = run_epoch(test_loader, model, model_org, criterion, None, DEVICE, False, training_type, feedback_type, desc="Testing")
        
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Accuracy: {accuracy_score(test_labels, test_preds):.4f}")
        print(f"F1: {f1_score(test_labels, test_preds, average='weighted'):.4f}")
        print(f"Kappa: {cohen_kappa_score(test_labels, test_preds, weights='quadratic'):.4f}")
        print(classification_report(test_labels, test_preds, target_names=[f"KL {i}" for i in range(NUM_CLASSES)]))
        results = []
        results.append(f"Test Loss: {test_loss:.4f}")
        results.append(f"Accuracy: {accuracy_score(test_labels, test_preds):.4f}")
        results.append(f"F1: {f1_score(test_labels, test_preds, average='weighted'):.4f}")
        results.append(f"Kappa: {cohen_kappa_score(test_labels, test_preds, weights='quadratic'):.4f}")
        results.append("\n" + classification_report(test_labels, test_preds, target_names=[f"KL {i}" for i in range(NUM_CLASSES)]))

        # print to console
        for line in results:
            print(line)

        # write to file
        save_path = os.path.join(IMG_SAVE_PATH, "inference_result.txt")
        with open(save_path, "w") as f:
            for line in results:
                f.write(line + "\n")
        ConfusionMatrixDisplay.from_predictions(test_labels, test_preds, normalize="true", cmap=plt.cm.Greens, values_format='.2f')
        plt.savefig(os.path.join(IMG_SAVE_PATH, f"cm_{training_type}.eps"), format='eps')
        plt.savefig(os.path.join(IMG_SAVE_PATH, f"cm_{training_type}.png"), format='png')
    else:
        print(f"Pretrained model not found at: {PRETRAINED_MODEL_PATH}")

    # 8. grad-cam visualization (choose one figure)
    # target_id = "9932578"
    # target_side = "R"
    # index = np.where(np.array(test_pids)==target_id + "_" + target_side)[0].item()
    # target_layer = [model.patch_feature_extractor.conv_block3[0]]
    # patches_test, label, id, feature = test_ds.__getitem__(index)
    # patch_bag_tensor = torch.stack(patches_test).to(DEVICE)  # shape: [41, 1, 16, 16]
    # print(test_pids[index], label)
    # model.eval()
    # logits, att_scores = model([patch_bag_tensor], model_org, attention_tool)

    # print(f"patch_bag_tensor shape: {patch_bag_tensor.shape}")  # shape you pass IN
    # print(f"logits shape: {logits.shape}")                      # shape OUT
    # print(f"att_scores shape: {att_scores.shape}")              # if relevant

    # # Argmax across classes
    # target_classes = logits.argmax(dim=1)
    # print(f"target_classes shape: {target_classes.shape}")      # should be [batch_size]

    # # If you want just the first class for score:
    # target_class = target_classes[0].item()
    # print(f"target_class: {target_class}")

    # # Use the first logit row (batch item 0) and its predicted class
    # score = logits[0, target_class]
    # print(f"score shape: {score.shape}")  # should be scalar, so shape = []

    # model.zero_grad()
    # score.backward(retain_graph=True)
    # process_CAM(model, target_layer, target_class, patch_bag_tensor, patches_test, training_type)

    # still need to recover it to original image


    # False negatives for class 1 predicted as 0: [np.str_('9322401_R'), np.str_('9697342_L')]


