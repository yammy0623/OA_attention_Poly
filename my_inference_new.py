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
from model import CompleteMILModel
import matplotlib.pyplot as plt

# ---------------- Configuration ---------------- #
H5_FILE = rf"original_data\V00\knee_patches_patient_grouped_16_100_px.h5"
CHECKPOINT_DIR = rf"original_data\V00\model_checkpoints"
MEAN_STD_FILE_PATH = os.path.join(CHECKPOINT_DIR, "mean_std_train_patches.npy")
PRETRAINED_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model_val_kappa.pth")

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
    interpolation_method='nearest'
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
    plt.savefig(r".\inference\heatmap.eps", format='eps')
    plt.savefig(r".\inference\heatmap.png", format='png')

def process_CAM(model, target_layer, target_class, patch_bag_tensor, patches_test):
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
            base_heatmap_alpha=0.3 # Explicitly setting the alpha from your original code
        )
def enhance_patches_with_heatmaps(
    patches_list,
    heatmaps_list,
    enhancement_strength=0.5
):
    """
    Enhance image patches by applying their corresponding heatmaps
    as feature boosts.

    Args:
        patches_list (list/np.array): List of image patches (each a NumPy array).
        heatmaps_list (list/np.array): List of heatmaps corresponding to patches.
        enhancement_strength (float): How much the heatmap boosts the patch.

    Returns:
        List of enhanced patch arrays.
    """
    if not patches_list:
        print("No patches to process.")
        return []

    if len(patches_list) != len(heatmaps_list):
        print("Error: Mismatch in lengths.")
        return []

    enhanced_patches_list = []
    for idx in range(len(patches_list)):
        patch = patches_list[idx].astype(np.float32)
        heatmap = heatmaps_list[idx].astype(np.float32)

        # Normalize heatmap to [0, 1]
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.ptp() + 1e-8)

        # Apply feature boost: element-wise multiplication or addition
        enhanced_patch = patch + enhancement_strength * heatmap_norm * patch

        # Clip to valid image range (assuming [0, 1])
        enhanced_patch = np.clip(enhanced_patch, 0, 1)

        enhanced_patches_list.append(enhanced_patch)

    return enhanced_patches_list

def run_epoch_CAM_1(method, loader, model, criterion, optimizer, device, is_training, desc=""):
    model.train() if is_training else model.eval()
    total_loss, all_preds, all_labels, num_samples = 0.0, [], [], 0
    target_layer = [model.patch_feature_extractor.conv_block3[0]]
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
    

    for bags, labels in tqdm(loader, desc=desc, leave=False):
        if not bags:
            continue

        moved_bags = [] # [16, 41, 1, 16, 16]
        valid_indices = []
        enhanced_patches_list = []
        for i, bag in enumerate(bags):
            print("one bag shape:", bag.to(device).shape)
            logits, att_scores = model(bag.to(device))
            target_class = logits.argmax(dim=1).item()
            batch_size = bag.shape[0]
            targets = [ ClassifierOutputTarget(target_class) ] * batch_size
            patch_bag_tensor = bag.to(device)
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
            grayscale_ensemble = None
            for item in cam_data_sources:
                if grayscale_ensemble is None:
                    grayscale_ensemble = 1 / len(cam_data_sources) * cam_data_sources[item]
                else:
                    grayscale_ensemble = grayscale_ensemble + 1 / len(cam_data_sources) * cam_data_sources[item]

            cam_data_sources["ensemble"] = grayscale_ensemble
            for method_name, heatmaps in cam_data_sources.items():
                if method_name == method:
                    enhanced_patches = enhance_patches_with_heatmaps(
                        patches_list=moved_bags,
                        heatmaps_list=heatmaps,
                        enhancement_strength=0.5
                    )
                else:
                    continue
            
            if bag.nelement() > 0:
                moved_bags.append(bag.to(device))
                valid_indices.append(i)
            with torch.set_grad_enabled(is_training):
                outputs, _ = model(enhanced_patches_list)
                if outputs.size(0) != labels.size(0):
                    print(f"Skipping batch due to shape mismatch: {outputs.shape} vs {labels.shape}")
                    continue

                loss = criterion(outputs, labels)
                if is_training:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            num_samples += labels.size(0)
        if not moved_bags:
            continue

        labels = labels[valid_indices].to(device)

    avg_loss = total_loss / num_samples if num_samples else 0
    return avg_loss, all_labels, all_preds

def run_epoch_CAM(method, loader, model, criterion, optimizer, device, is_training, desc=""):
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM, LayerCAM

    model.train() if is_training else model.eval()

    total_loss, all_preds, all_labels, num_samples = 0.0, [], [], 0

    target_layer = [model.patch_feature_extractor.conv_block3[0]]
    cam_methods = {
        "GradCAM": GradCAM(model=model.patch_feature_extractor, target_layers=target_layer),
        "GradCAM++": GradCAMPlusPlus(model=model.patch_feature_extractor, target_layers=target_layer),
        "ScoreCAM": ScoreCAM(model=model.patch_feature_extractor, target_layers=target_layer),
        "AblationCAM": AblationCAM(model=model.patch_feature_extractor, target_layers=target_layer),
        "LayerCAM": LayerCAM(model=model.patch_feature_extractor, target_layers=target_layer)
    }

    cam = cam_methods[method]

    for bags, labels in tqdm(loader, desc=desc, leave=False):
        if not bags:
            continue

        for i, bag in enumerate(bags):
            # bag: [N, C, H, W]
            bag = bag.to(device)
            print("Bag shape:", bag.shape)

            # Run original forward to get predicted class
            logits, _ = model(bag)
            target_class = logits.argmax(dim=1).item()

            targets = [ClassifierOutputTarget(target_class)] * bag.shape[0]

            # Generate heatmaps
            grayscale_cams = cam(input_tensor=bag, targets=targets)

            # Enhance
            bag_np = bag.detach().cpu().numpy()
            enhanced_patches = enhance_patches_with_heatmaps(
                patches_list=bag_np,
                heatmaps_list=grayscale_cams,
                enhancement_strength=0.5
            )
            # Convert back to tensor
            enhanced_bag = torch.stack(
                [torch.from_numpy(p) for p in enhanced_patches]
            ).float().to(device)

            # Forward pass on enhanced patches
            with torch.set_grad_enabled(is_training):
                outputs, _ = model(enhanced_bag)

                if outputs.size(0) != labels.size(0):
                    print(f"Skipping batch due to shape mismatch: {outputs.shape} vs {labels.shape}")
                    continue

                labels = labels.to(device)
                loss = criterion(outputs, labels)

                if is_training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * labels.size(0)

                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                num_samples += labels.size(0)

    avg_loss = total_loss / num_samples if num_samples else 0
    return avg_loss, all_labels, all_preds

def run_epoch(loader, model, criterion, optimizer, device, is_training, desc=""):
    model.train() if is_training else model.eval()
    total_loss, all_preds, all_labels, num_samples = 0.0, [], [], 0

    for bags, labels in tqdm(loader, desc=desc, leave=False):
        if not bags:
            continue

        moved_bags = []
        valid_indices = []
        for i, bag in enumerate(bags):
            if bag.nelement() > 0:
                moved_bags.append(bag.to(device))
                valid_indices.append(i)

        if not moved_bags:
            continue

        labels = labels[valid_indices].to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs, _ = model(moved_bags)
            if outputs.size(0) != labels.size(0):
                print(f"Skipping batch due to shape mismatch: {outputs.shape} vs {labels.shape}")
                continue

            loss = criterion(outputs, labels)
            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        num_samples += labels.size(0)

    avg_loss = total_loss / num_samples if num_samples else 0
    return avg_loss, all_labels, all_preds

# ---------------- Main Execution ---------------- #
if __name__ == '__main__':
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
    # MIL extractor
    model = CompleteMILModel(FEATURE_EXTRACTOR_OUT_DIM, NUM_CLASSES, AGGREGATION_TYPE).to(DEVICE)
    with h5py.File(H5_FILE, 'r') as hf:
        train_grades = [hf[group]['kl_grade'][0] for group in train_ds.sample_group_names]
    class_counts = np.bincount(train_grades, minlength=NUM_CLASSES)
    weights = 1.0 / (class_counts + 1e-6)
    weights /= weights.sum()
    class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    print(f"Class Weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    # 7. Test Inference
    WITHCAM = True
    cam_data_sources = ["GradCAM", "GradCAM++", "ScoreCAM", "AblationCAM", "LayerCAM", "ensemble"]

    if os.path.exists(PRETRAINED_MODEL_PATH):
        if WITHCAM:
            for method in cam_data_sources:
                model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
                test_loss, test_labels, test_preds = run_epoch_CAM(method, test_loader, model, criterion, None, DEVICE, False, desc="Testing")
                print("Method: ", method)
                print(f"\nTest Loss: {test_loss:.4f}")
                print(f"Accuracy: {accuracy_score(test_labels, test_preds):.4f}")
                print(f"F1: {f1_score(test_labels, test_preds, average='weighted'):.4f}")
                print(f"Kappa: {cohen_kappa_score(test_labels, test_preds, weights='quadratic'):.4f}")
                print(classification_report(test_labels, test_preds, target_names=[f"KL {i}" for i in range(NUM_CLASSES)]))
                ConfusionMatrixDisplay.from_predictions(test_labels, test_preds, normalize="true", cmap=plt.cm.Greens, values_format='.2f')
                plt.savefig(f".\inference\cm_{method}.eps", format='eps')
                plt.savefig(f".\inference\cm_{method}.png", format='png')
        else:
            model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
            test_loss, test_labels, test_preds = run_epoch(test_loader, model, criterion, None, DEVICE, False, desc="Testing")
            print(f"\nTest Loss: {test_loss:.4f}")
            print(f"Accuracy: {accuracy_score(test_labels, test_preds):.4f}")
            print(f"F1: {f1_score(test_labels, test_preds, average='weighted'):.4f}")
            print(f"Kappa: {cohen_kappa_score(test_labels, test_preds, weights='quadratic'):.4f}")
            print(classification_report(test_labels, test_preds, target_names=[f"KL {i}" for i in range(NUM_CLASSES)]))
            ConfusionMatrixDisplay.from_predictions(test_labels, test_preds, normalize="true", cmap=plt.cm.Greens, values_format='.2f')
            plt.savefig(r".\inference\cm.eps", format='eps')
            plt.savefig(r".\inference\cm.png", format='png')
    else:
        print(f"Pretrained model not found at: {PRETRAINED_MODEL_PATH}")
    
    

    # 8. grad-cam visualization (choose one figure)
    target_id = "9932578"
    target_side = "R"
    index = np.where(np.array(test_pids)==target_id + "_" + target_side)[0].item()
    target_layer = [model.patch_feature_extractor.conv_block3[0]]
    patches_test, label = test_ds.__getitem__(index)
    patch_bag_tensor = torch.stack(patches_test).to(DEVICE)  # shape: [41, 1, 16, 16]
    print(test_pids[index], label)
    model.eval()
    logits, att_scores = model([patch_bag_tensor])
    target_class = logits.argmax(dim=1).item()
    score = logits[0, target_class]
    model.zero_grad()
    score.backward(retain_graph=True)
    # process_CAM(model, target_layer, target_class, patch_bag_tensor, patches_test)