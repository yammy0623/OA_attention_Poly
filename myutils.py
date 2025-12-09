import os
import numpy as np
import h5py
from tqdm import tqdm
import torchvision.transforms as transforms
from data_augmentation import CorrectBrightness, CorrectContrast, CorrectGamma
import torch
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report, ConfusionMatrixDisplay
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, LayerCAM
import torch.nn as nn
import matplotlib.pyplot as plt
import pydicom
import matplotlib.patches as patches
import math
from losses import CoralLossWeighted, CoralLossEffective, CoralFocalLoss, CoralFocalLoss_MultiTask, CoralLoss_MultiTask, CrossEntropy_MultiTask, CoralFocalLoss_MultiTask_MetricsBalanced
from model import (
    CompleteMILModel,
    CompleteMILModel_MultiTask,
    CompleteMILModel_MultiTask_SharedHead,
    CompleteMILOrdinalModel,
    CompleteMILOrdinal_MultiTask_Model,
    CompleteMILCoral_MultiTask_Model,
    CompleteMILModel_wGP_MultiTask,
    CompleteMILModel_MultiTask_imedslab,

)

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


def labels_to_levels(labels, num_classes):
    """
    Convert integer labels to CORAL level vectors.
    
    labels: (N,) int tensor with values 0..K-1
    num_classes: K
    returns: (N, K-1) float tensor
    """
    N = labels.shape[0]
    levels = torch.zeros((N, num_classes-1), device=labels.device)
    for k in range(num_classes-1):
        levels[:, k] = (labels > k).float()
    return levels



def calculate_mean_std(h5_file, sample_groups, save_path, DEFAULT_MAX_PIXEL_VALUE):
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

# Compute class weights
def compute_class_weights(classweight_type, counts, beta=0.9999, device="cuda"):
    if classweight_type== "effective":  # effective class weights
        weights = compute_effective_class_weights(counts, num_classes=len(counts), beta=beta)
        weights = np.array(weights)
        # Optional scaling for specific classes
        weights[1, :] = weights[1, :] * 2.0
        return torch.tensor(weights, dtype=torch.float).to(device)
    elif classweight_type== "inv":  # inverse frequency weights
        weights = 1.0 / (counts + 1e-6)  # Add epsilon
        weights = weights / np.sum(weights) * len(counts)  # Normalize to num_classes
        return torch.tensor(weights, dtype=torch.float).to(device)
    else:
        return None

def compute_multiTask_class_weights(classweight_type, task_counts, beta=0.9999, device="cuda"):
    """
    Args:
        task_counts: dict, task_name -> counts array (num_classes,)
        classweight_type: "effective" or "inv"
    Returns:
        dict of task_name -> class weights tensor
    """
    weights_dict = {}
    for task, counts in task_counts.items():
        if classweight_type == "effective":
            weights = compute_effective_class_weights(counts, num_classes=len(counts), beta=beta)
            weights = np.array(weights)
            # optional scaling for specific tasks (only if you want)
            if task == "kl":
                weights = weights * 2.0
            weights_dict[task] = torch.tensor(weights, dtype=torch.float).to(device)
        elif classweight_type == "inv":
            weights = 1.0 / (counts + 1e-6)
            weights = weights / np.sum(weights) * len(counts)
            weights_dict[task] = torch.tensor(weights, dtype=torch.float).to(device)
        else:
            weights_dict[task] = None
    return weights_dict


    

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



def get_criterion(lossfcn_type, class_weights_tensor, oai_task_num_classes=None):
    if lossfcn_type == "CoralLossWeighted":
        return CoralLossWeighted(class_weights=class_weights_tensor)
    elif lossfcn_type == "CoralFocalLoss":
        return CoralFocalLoss(class_weights=class_weights_tensor, gamma=2.0, alpha=0.25)
    elif lossfcn_type == "CoralLossEffective":
        return CoralLossEffective(threshold_weights=class_weights_tensor)
    elif lossfcn_type == "CoralFocalLoss_MultiTask":
        return CoralFocalLoss_MultiTask(oai_task_num_classes, is_learn_task_weights=True, class_weights=class_weights_tensor)
    elif lossfcn_type == "CoralLoss_MultiTask":
        return CoralLoss_MultiTask(reduction='mean', class_weights=class_weights_tensor)
    elif lossfcn_type == "CrossEntropy":
        return nn.CrossEntropyLoss(weight=class_weights_tensor)
    elif lossfcn_type == "CrossEntropy_MultiTask":
        return CrossEntropy_MultiTask(class_weights=class_weights_tensor)
    elif lossfcn_type == "CoralFocalLoss_MultiTask_MetricsBalanced":
        return CoralFocalLoss_MultiTask_MetricsBalanced(oai_task_num_classes, is_learn_task_weights=True, class_weights=class_weights_tensor)
    elif lossfcn_type == "BCEWithLogitsLoss_MultiTask":
        from losses import BCEWithLogitsLoss_MultiTask
        return BCEWithLogitsLoss_MultiTask(class_weights=class_weights_tensor)
    else:
        print("No criterion")


def get_model(config):
    if config.model_type == "MILOrdinal":  # ordinal model
        model = CompleteMILOrdinalModel(config.FEATURE_EXTRACTOR_OUT_DIM, config.KL_NUM_CLASSES, config.AGGREGATION_TYPE).to(config.DEVICE)
    elif config.model_type == "MILOrdinal_MultiTask":  # multitask
        model = CompleteMILOrdinal_MultiTask_Model(config.FEATURE_EXTRACTOR_OUT_DIM,
                                                   config.OARSI_TASKS,
                                                   config.AGGREGATION_TYPE).to(config.DEVICE)
    elif config.model_type == "MILCoral_MultiTask":
        model = CompleteMILCoral_MultiTask_Model(config.FEATURE_EXTRACTOR_OUT_DIM,
                                                config.OARSI_TASKS,
                                                config.AGGREGATION_TYPE).to(config.DEVICE)
    elif config.model_type == "MIL": 
        model = CompleteMILModel(config.FEATURE_EXTRACTOR_OUT_DIM,
                                     config.KL_NUM_CLASSES,
                                     config.AGGREGATION_TYPE).to(config.DEVICE)
    elif config.model_type == "MIL_MultiTask":
        model = CompleteMILModel_MultiTask(config.FEATURE_EXTRACTOR_OUT_DIM,
                                     config.OARSI_TASKS,
                                     config.AGGREGATION_TYPE).to(config.DEVICE)
    elif config.model_type == "MIL_MultiTask_SharedHead":
        model = CompleteMILModel_MultiTask_SharedHead(config.FEATURE_EXTRACTOR_OUT_DIM,
                                     config.OARSI_TASKS,
                                     config.AGGREGATION_TYPE).to(config.DEVICE)
    elif config.model_type == "MIL_wGP_MultiTask":
        model = CompleteMILModel_wGP_MultiTask(config.FEATURE_EXTRACTOR_OUT_DIM,
                                     config.OARSI_TASKS,
                                     config.AGGREGATION_TYPE).to(config.DEVICE)
    elif config.model_type == "MIL_MultiTask_imedslab":
        from model import CompleteMILModel_MultiTask_imedslab
        model = CompleteMILModel_MultiTask_imedslab(config.FEATURE_EXTRACTOR_OUT_DIM,
                                     config.OARSI_TASKS,
                                     config.AGGREGATION_TYPE).to(config.DEVICE)
    elif config.model_type == "MILOrdinal_MultiTask_imedslab":
        from model import CompleteMILOrdinalModel_MultiTask_imedslab
        model = CompleteMILOrdinalModel_MultiTask_imedslab(config.FEATURE_EXTRACTOR_OUT_DIM,
                                     config.OARSI_TASKS,
                                     config.AGGREGATION_TYPE).to(config.DEVICE)
    return model


def get_model_org(config):
    if config.feedback_type == "off":
        model_org = None
    else:
        model_org = CompleteMILModel(config.FEATURE_EXTRACTOR_OUT_DIM,
                                     config.KL_NUM_CLASSES,
                                     config.AGGREGATION_TYPE).to(config.DEVICE)
        model_org.load_state_dict(torch.load(config.PRETRAINED_MODEL_PATH, map_location=config.DEVICE))
    return model_org

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
    method_name="GradCAM",
    img_save_path="",
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
    plt.savefig(os.path.join(img_save_path, f"heatmap_vis_{method_name}.eps"), format='eps')
    plt.savefig(os.path.join(img_save_path, f"heatmap_vis_{method_name}.png"), format='png')


def process_CAM(model, target_layer, target_class, patch_bag_tensor, patches_test, img_save_path):
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

    cam_data_sources["Ensemble"] = grayscale_ensemble

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
            method_name=method_name,
            img_save_path=img_save_path
        )

    return cam_data_sources, PATCH_POINT_INDICES

def create_redsalpha():
    from matplotlib.colors import LinearSegmentedColormap
    ncolors = 256
    base = plt.get_cmap('Reds')(np.arange(ncolors))

    # 2. set alpha from 0 (white) → 1 (red)
    idx = np.arange(ncolors)
    v = idx / (ncolors - 1)
    alpha = np.where(v >= 0.5, 1.0, 2 * v)
    base[:, -1] = alpha

    # 3. create a new colormap
    reds_alpha = LinearSegmentedColormap.from_list('Reds_alpha', base)
    return reds_alpha

# Adapt from Tiulpin et al. (2018)
def process_xray(img, cut_min=5, cut_max=99, multiplier=255):
    """
    This function changes the histogram of the image by doing global contrast normalization
    
    Parameters
    ----------
    img : array_like
        Image
    cut_min : int
         Low percentile to trim
    cut_max : int
        Highest percentile trim
    multiplier : int
        Multiplier to apply after global contrast normalization
        
    Returns
    -------
    array_like
        Returns a processed image
    
    """

    img = img.copy()
    lim1, lim2 = np.percentile(img, [cut_min, cut_max])
    img[img < lim1] = lim1
    img[img > lim2] = lim2

    img -= lim1
    img /= img.max()
    img *= multiplier
    return img

def patchFromPoint(point, size):
    '''
    point : the landmark
    size  : half of the length of the box (square)
    '''
    topLeft = (int(point[0] - size), int(point[1] - size))
    botRight = (int(point[0] + size), int(point[1] + size))
    return topLeft, botRight

def visualize_attention_on_img(
    save_path,
    file_path,
    patient_id,
    index_all,
    shapes_L_2d,
    shapes_R_2d,
    att_scores,
    side,
    patchFromPoint,
    process_xray,
    title=None
):
    # Load DICOM and preprocess image
    dicom_data = pydicom.dcmread(file_path)
    image = np.frombuffer(dicom_data.pixel_array.copy(), dtype=np.uint16).copy().astype(np.float64)
    image = process_xray(image, 5, 99, 65535)
    image = image.reshape((dicom_data.Rows, dicom_data.Columns))

    # Set up visualization parameters
    cmap = plt.get_cmap('Reds')
    size = np.sqrt((100 * 100) / (3560 * 4320)) * np.sqrt(image.shape[0] * image.shape[1]) / 2

    # Create figure
    fig, ax = plt.subplots(figsize=(35, 43))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    ax.set_title(title or f"{dicom_data.SeriesDescription} ({patient_id})")

    # Draw landmark points
    ax.scatter(shapes_L_2d[index_all][:, 0], shapes_L_2d[index_all][:, 1], c='red')
    ax.scatter(shapes_R_2d[index_all][:, 0], shapes_R_2d[index_all][:, 1], c='red')
    for i_pt in range(len(shapes_L_2d[index_all])):
        ax.annotate(str(i_pt), shapes_L_2d[index_all][i_pt], color='black')
        ax.annotate(str(i_pt + len(shapes_L_2d[index_all])),
                    shapes_R_2d[index_all][i_pt], color='black')

    # Define patch indices and overlay patches
    range1 = np.arange(9, 27)
    range2 = np.arange(44, 67)
    PATCH_POINT_INDICES = np.concatenate([range1, range2])

    for id, i in enumerate(PATCH_POINT_INDICES):
        w = att_scores[id]
        rgba = cmap(w)

        # Left knee
        topLeft, botRight = patchFromPoint(shapes_L_2d[index_all][i], size=size)
        rectL = patches.Rectangle(
            (topLeft[0], topLeft[1]),
            botRight[0] - topLeft[0],
            botRight[1] - topLeft[1],
            linewidth=1,
            edgecolor='none' if side == "L" else 'black',
            facecolor=rgba if side == "L" else 'none',
            alpha=0.4 * w if side == "L" else 1
        )
        ax.add_patch(rectL)

        # Right knee
        topLeft, botRight = patchFromPoint(shapes_R_2d[index_all][i], size=size)
        rectR = patches.Rectangle(
            (topLeft[0], topLeft[1]),
            botRight[0] - topLeft[0],
            botRight[1] - topLeft[1],
            linewidth=1,
            edgecolor='none' if side == "R" else 'black',
            facecolor=rgba if side == "R" else 'none',
            alpha=0.4 * w if side == "R" else 1
        )
        ax.add_patch(rectR)

    # plt.show()
    plt.savefig(os.path.join(save_path, f"attention_{patient_id}_{side}.png"), format='png')

def visualize_cam_comparisons(
    save_path,
    patient_id,
    index_all,
    index_test,
    side,
    test_labels,
    test_preds,
    att_scores,
    grayscale_cam_dict,
    process_xray_func,
    patch_from_point_func,
    shapes_L_2d,
    shapes_R_2d,
    file_path_template,
    patch_point_indices,
    cmap_obj,
    alpha_val=0.8,
    subplot_width=10,
    subplot_height=12
):
    """
    Visualizes and saves CAM comparison figure for one patient.

    Args:
        patient_id (str): Current patient ID
        index_all (int): Index into shapes for current patient
        index_test (int): Index for prediction/KL info
        side (str): "L" or "R"
        test_kls (List[int]): True KL grades
        test_preds (np.ndarray): Model prediction probabilities
        att_scores (List[float]): Attention weights for patches
        grayscale_cam_dict (dict): Dict of {method_name: list of heatmaps}
        process_xray_func (callable): Your image preprocessing function
        patch_from_point_func (callable): Your patch coordinate function
        shapes_L_2d (np.ndarray): Left knee landmark coordinates
        shapes_R_2d (np.ndarray): Right knee landmark coordinates
        file_path_template (str): Template path to load DICOMs with {patient_id}
        patch_point_indices (list[int]): Indices of patch points
        cmap_obj (matplotlib.colors.Colormap): Colormap to use for CAM
        output_dir (str): Directory to save figures
        alpha_val (float): CAM overlay transparency
        subplot_width (int): Width of each subplot
        subplot_height (int): Height of each subplot
    """
    import pydicom
    import cv2
    from scipy.ndimage import gaussian_filter

    def load_and_preprocess_dicom(file_path_template, patient_id, process_xray_func):
        file_path = file_path_template.format(patient_id=patient_id)
        dicom_data = pydicom.dcmread(file_path)
        image = dicom_data.pixel_array.copy().astype(np.float64)
        image = process_xray_func(image, 5, 99, 65535)
        image = image.reshape((dicom_data.Rows, dicom_data.Columns))
        return dicom_data, image

    def calculate_patch_sizes(image_shape):
        ref_patch_dim = 100
        ref_img_area = 3560 * 4320
        img_area = image_shape[0] * image_shape[1]
        size_factor = (ref_patch_dim / np.sqrt(ref_img_area)) * np.sqrt(img_area) / 2.0
        patch_dim = round(size_factor * 2)
        return size_factor, (patch_dim, patch_dim)

    # Import internal plotting logic
    from matplotlib import patches
    def plot_heatmaps_on_ax(
        ax, dicom_data, base_image, shapes_L_current, shapes_R_current,
        patch_point_indices, current_grayscale_cams, current_att_scores,
        patch_half_size_val, patch_dims_tuple_val, cmap_obj, base_alpha_val,
        target_side_focus, patch_from_point_func, method_name=""
    ):
        hh_img, ww_img = base_image.shape
        active_side_shapes = shapes_L_current if target_side_focus == "L" else shapes_R_current

        if active_side_shapes.size > 0:
            min_coords = np.min(active_side_shapes, axis=0)
            max_coords = np.max(active_side_shapes, axis=0)
            padding_amount = 2 * patch_dims_tuple_val[0]
            w_min_zoomed = min_coords[0] - padding_amount
            h_min_zoomed = min_coords[1] - padding_amount
            w_max_zoomed = max_coords[0] + padding_amount
            h_max_zoomed = max_coords[1] + padding_amount
        else:
            w_min_zoomed, h_min_zoomed = 0, 0
            w_max_zoomed, h_max_zoomed = ww_img, hh_img

        ax.imshow(base_image, cmap='gray', origin='upper', extent=(0, ww_img, hh_img, 0), zorder=0)
        ax.set_xlim(max(0, w_min_zoomed), min(ww_img, w_max_zoomed))
        ax.set_ylim(min(hh_img, h_max_zoomed), max(0, h_min_zoomed))
        ax.axis('off')
        ax.set_title(method_name, fontsize=25)

        summed_heatmap_canvas = np.zeros_like(base_image, dtype=np.float64)

        for id_patch, point_idx in enumerate(patch_point_indices):
            current_shapes = shapes_L_current if target_side_focus == "L" else shapes_R_current
            if point_idx >= current_shapes.shape[0]:
                continue

            cam_patch = current_grayscale_cams[id_patch]
            att_w = current_att_scores[id_patch]

            center = current_shapes[point_idx]
            flip = (target_side_focus == "L")

            resized = cv2.resize(cam_patch, patch_dims_tuple_val, interpolation=cv2.INTER_LINEAR)
            if flip:
                resized = np.fliplr(resized)

            topLeft, botRight = patch_from_point_func(center, size=patch_half_size_val)
            x0, y0 = int(round(topLeft[0])), int(round(topLeft[1]))
            x1, y1 = int(round(botRight[0])), int(round(botRight[1]))

            if x1 <= x0 or y1 <= y0:
                continue

            if resized.shape != (y1 - y0, x1 - x0):
                resized = cv2.resize(resized, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)

            # Clip to canvas
            x0_clip, x1_clip = max(0, x0), min(ww_img, x1)
            y0_clip, y1_clip = max(0, y0), min(hh_img, y1)

            h_slice = resized[(y0_clip - y0):(y1_clip - y0), (x0_clip - x0):(x1_clip - x0)]
            summed_heatmap_canvas[y0_clip:y1_clip, x0_clip:x1_clip] += h_slice * att_w

        min_val, max_val = np.min(summed_heatmap_canvas), np.max(summed_heatmap_canvas)
        if max_val > min_val:
            normed = (summed_heatmap_canvas - min_val) / (max_val - min_val)
        else:
            normed = np.zeros_like(summed_heatmap_canvas)
        smoothed = gaussian_filter(normed, sigma=(patch_dims_tuple_val[0] - 1) / 8)
        ax.imshow(smoothed, cmap=cmap_obj, alpha=base_alpha_val, extent=(0, ww_img, hh_img, 0), origin='upper', zorder=1)

        return normed

    # ---- Start execution ----
    dicom_data, base_image = load_and_preprocess_dicom(file_path_template, patient_id, process_xray_func)
    patch_half_size, patch_dims = calculate_patch_sizes(base_image.shape)
    shapes_L_current = shapes_L_2d[index_all]
    shapes_R_current = shapes_R_2d[index_all]

    # Ensemble CAM (optional)
    method_dict = grayscale_cam_dict.copy()
    # if method_dict:
    #     keys = list(method_dict.keys())
    #     n_methods = len(keys)
    #     ensemble = [np.zeros_like(method_dict[keys[0]][i], dtype=np.float64) for i in range(len(method_dict[keys[0]]))]
    #     for method_cams in method_dict.values():
    #         if len(method_cams) != len(ensemble):
    #             continue
    #         for i in range(len(method_cams)):
    #             ensemble[i] += method_cams[i] / n_methods
    #     method_dict["Ensemble"] = ensemble

    n_methods = len(method_dict)
    n_cols = 3
    n_rows = math.ceil(n_methods / n_cols)
    fig_width = subplot_width * n_cols
    fig_height = subplot_height * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    # fig.suptitle(
    #     f"{dicom_data.SeriesDescription} - CAM Method Comparison:\n{patient_id}_{side} [{index_test}] (True: KL-{test_labels['kl'[index_test]]}, Pred: KL-{test_preds['kl'][index_test].argmax()})",
    #     fontsize=30, y=0.99
    # )

    true_kl = test_labels["kl"][index_test]
    pred_kl = test_preds["kl"][index_test]

    # OARSI true/pred for each task
    true_jsnm = test_labels["jsnm"][index_test]
    pred_jsnm = test_preds["jsnm"][index_test]

    true_jsnl = test_labels["jsnl"][index_test]
    pred_jsnl = test_preds["jsnl"][index_test]

    if "ostm" in test_labels:
        true_osfm = test_labels["osfm"][index_test]
        pred_osfm = test_preds["osfm"][index_test]
        true_ostm = test_labels["ostm"][index_test]
        pred_ostm = test_preds["ostm"][index_test]
        true_ostl = test_labels["ostl"][index_test]
        pred_ostl = test_preds["ostl"][index_test]
        true_osfl = test_labels["osfl"][index_test]
        pred_osfl = test_preds["osfl"][index_test]

        fig.suptitle(
            f"{dicom_data.SeriesDescription} - CAM Method Comparison:\n"
            f"{patient_id}_{side} [{index_test}]\n"
            f"KL: True {true_kl}, Pred {pred_kl} | "
            f"JSN-M: T{int(true_jsnm)}/P{pred_jsnm}, "
            f"JSN-L: T{int(true_jsnl)}/P{pred_jsnl}, "
            f"OSF-M: T{int(true_osfm)}/P{pred_osfm}, "
            f"OST-M: T{int(true_ostm)}/P{pred_ostm}, "
            f"OST-L: T{int(true_ostl)}/P{pred_ostl}, "
            f"OSF-L: T{int(true_osfl)}/P{pred_osfl}",
            fontsize=22, y=0.99
        )
    else:
        fig.suptitle(
            f"{dicom_data.SeriesDescription} - CAM Method Comparison:\n"
            f"{patient_id}_{side} [{index_test}]\n"
            f"KL: True {true_kl}, Pred {pred_kl} | "
            f"JSN-M: T{int(true_jsnm)}/P{pred_jsnm}, "
            f"JSN-L: T{int(true_jsnl)}/P{pred_jsnl}",
            fontsize=30, y=0.99
        )

    for i, (method_name, cam_list) in enumerate(method_dict.items()):
        if i >= len(axes_flat): break
        print(f"Plotting: {method_name}")
        plot_heatmaps_on_ax(
            ax=axes_flat[i],
            dicom_data=dicom_data,
            base_image=base_image,
            shapes_L_current=shapes_L_current,
            shapes_R_current=shapes_R_current,
            patch_point_indices=patch_point_indices,
            current_grayscale_cams=cam_list,
            current_att_scores=att_scores,
            patch_half_size_val=patch_half_size,
            patch_dims_tuple_val=patch_dims,
            cmap_obj=cmap_obj,
            base_alpha_val=alpha_val,
            target_side_focus=side,
            patch_from_point_func=patch_from_point_func,
            method_name=method_name
        )

    for j in range(n_methods, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    # os.makedirs(output_dir, exist_ok=True)
    # save_path = os.path.join(output_dir, f"{patient_id}_{side}.png")
    fig.savefig(os.path.join(save_path, f"cam_{patient_id}_{side}.png"), format='png')

    # fig.savefig(save_path, dpi=150, bbox_inches='tight')
    # print(f"Saved to: {save_path}")
    # plt.show()


def normalize_attention_scores(scores):
    min_s = np.min(scores)
    max_s = np.max(scores)
    if max_s == min_s: # Avoid division by zero
        return np.zeros_like(scores) if min_s == 0 else np.ones_like(scores) * 0.5
    return (scores - min_s) / (max_s - min_s)


def run_gradcam_visualization(target_id, target_side, model, model_org, test_ds, test_pids,
                              process_CAM, process_xray, patchFromPoint,
                              reds_alpha, output_dir="./ecam"):

    DEVICE = next(model.parameters()).device

    # Step 1: 找到目標索引與模型輸出
    index = np.where(np.array(test_pids) == f"{target_id}_{target_side}")[0].item()
    target_layer = [model.patch_feature_extractor.conv_block3[0]]
    patches_test, label = test_ds.__getitem__(index)
    patch_bag_tensor = torch.stack(patches_test).to(DEVICE)

    model.eval()
    logits, att_scores = model([patch_bag_tensor], model_org)
    target_class = logits.argmax(dim=1)[0].item()
    score = logits[0, target_class]

    model.zero_grad()
    score.backward(retain_graph=True)
    grayscale_cam_dict, PATCH_POINT_INDICES = process_CAM(
        model, target_layer, target_class, patch_bag_tensor, patches_test
    )

    # Step 2: 載入 Ground Truth 資訊
    id_shapeLR_kl = np.load(r"./inference/id_shapeLR_V00.npz")
    test_id_pred = np.load(r"./inference/test_pred.npz")
    test_id_att = np.load(r"./inference/test_att_scores.npz")

    patient_ids = id_shapeLR_kl["id"]
    shapes_L_2d = id_shapeLR_kl["shapes_L"]
    shapes_R_2d = id_shapeLR_kl["shapes_R"]
    kl_grades_L_np = id_shapeLR_kl["KL_L"]
    kl_grades_R_np = id_shapeLR_kl["KL_R"]

    test_ids = test_id_pred["id"]
    test_preds = test_id_pred["prob"]
    test_kls = test_id_pred["true_kl"]
    test_att_scores = test_id_att["att_scores"]

    index_test = index
    pid, side = str.split(test_pids[index_test], "_")
    index_all = np.where(np.array(patient_ids) == pid)[0].item()

    print(f"Case: {test_ids[index_test]}, Ground Truth KL = {test_kls[index_test]}, Pred = {np.argmax(test_preds[index_test])}")
    print("Prediction Probs:", " ".join(f"{x:.4f}" for x in test_preds[index_test]))
    print("Correct!" if np.argmax(test_preds[index_test]) == test_kls[index_test] else "Wrong!")

    # Step 3: Attention 視覺化
    att_scores_norm = normalize_attention_scores(test_att_scores[index_test])
    file_path = rf"./inference/Bilateral_PA_Fixed_Flexion_Knee/{patient_ids[index_all]}.dcm"

    visualize_attention_on_img(
        file_path=file_path,
        patient_id=patient_ids[index_all],
        index_all=index_all,
        shapes_L_2d=shapes_L_2d,
        shapes_R_2d=shapes_R_2d,
        att_scores=att_scores_norm,
        side=side,
        patchFromPoint=patchFromPoint,
        process_xray=process_xray
    )

    # Step 4: GradCAM + Attention 比較視覺化
    visualize_cam_comparisons(
        patient_id=patient_ids[index_all],
        index_all=index_all,
        index_test=index_test,
        side=side,
        test_kls=test_kls,
        test_preds=test_preds,
        att_scores=att_scores_norm,
        grayscale_cam_dict=grayscale_cam_dict,
        process_xray_func=process_xray,
        patch_from_point_func=patchFromPoint,
        shapes_L_2d=shapes_L_2d,
        shapes_R_2d=shapes_R_2d,
        file_path_template=rf"./inference/Bilateral_PA_Fixed_Flexion_Knee/{{patient_id}}.dcm",
        patch_point_indices=PATCH_POINT_INDICES,
        cmap_obj=reds_alpha,
        output_dir=output_dir
    )
