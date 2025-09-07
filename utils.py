import numpy as np
import h5py
from tqdm import tqdm


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

import torchvision.transforms as transforms
from utils import CorrectBrightness, CorrectContrast, CorrectGamma
import torch

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


# Compute class weights
def compute_class_weights(counts, feedback_type=None, beta=0.9999, device="cuda"):
    if feedback_type == 13:  # effective class weights
        weights = compute_effective_class_weights(counts, num_classes=len(counts), beta=beta)
        weights = np.array(weights)
        # Optional scaling for specific classes
        weights[1, :] = weights[1, :] * 2.0
    else:  # inverse frequency weights
        weights = 1.0 / (counts + 1e-6)  # Add epsilon
        weights = weights / np.sum(weights) * len(counts)  # Normalize to num_classes

    return torch.tensor(weights, dtype=torch.float).to(device)