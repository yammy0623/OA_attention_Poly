import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Example: Generate synthetic high-dimensional data
from sklearn.datasets import load_digits
import torch



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
from model import CompleteMILModel, CompleteMILCamModel, CompleteMILCamModel_Attention_feedback
import matplotlib.pyplot as plt
import argparse
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, LayerCAM

# ---------------- Configuration ---------------- #
H5_FILE = os.path.join("model_checkpoints_tnc_final", "knee_patches_patient_grouped_16_100.h5")
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--training_type", type=str, default="original", choices=["original", "GradCAM", "GradCAMPlusPlus", "ScoreCAM", "AblationCAM", "LayerCAM"])
    parser.add_argument("--pre_ckpt", type=str, default="PRE_CHECKPOINT_DIR")
    parser.add_argument("--current_ckpt", type=str, default="CHECKPOINT_DIR")
    parser.add_argument("--feedback_type", type=int, default=1, choices=[1, 2, 3, 4, 5], help="1: (w*cam*p + p), 2: (w(cam*p + p)), 3: (w*cam*p), 4: (w*cam*p + p), 5: (w*cam*p + w*p)")


    
    args = parser.parse_args()
    training_type = args.training_type
    PRE_CHECKPOINT_DIR = args.pre_ckpt
    CHECKPOINT_DIR = args.current_ckpt
    CHECKPOINT_DIR = os.path.join("original_data", "V00", CHECKPOINT_DIR)
    feedback_type = args.feedback_type
    IMG_SAVE_PATH=CHECKPOINT_DIR


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
    # print(f"First test PIDs: {test_pids[0]}")
    # cc

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
    train_ds = KneeMILDataset(H5_FILE, train_pids, transform=train_transform)
    val_ds = KneeMILDataset(H5_FILE, val_pids, transform=val_transform)
    test_ds = KneeMILDataset(H5_FILE, test_pids, transform=val_transform)

    train_loader = DataLoader(train_ds, BATCH_SIZE, True, collate_fn=mil_collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, BATCH_SIZE, False, collate_fn=mil_collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_ds, BATCH_SIZE, False, collate_fn=mil_collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # for i, (bags, labels, ids) in enumerate(test_loader):
    #     if not bags:
    #         continue
    #     print(f"Batch {i}: {len(bags)} bags, {labels.shape[0]} labels")
    #     break


    # 6. Model, Loss, Optimizer
    model = CompleteMILCamModel_Attention_feedback(FEATURE_EXTRACTOR_OUT_DIM, NUM_CLASSES, training_type, feedback_type, AGGREGATION_TYPE).to(DEVICE)
    model_org = CompleteMILModel(FEATURE_EXTRACTOR_OUT_DIM, NUM_CLASSES, AGGREGATION_TYPE).to(DEVICE)
    
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
    target_layer = [model_org.patch_feature_extractor.conv_block3[0]]
    gradcam_type = training_type
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
    is_training = False  # Set to False for inference mode
    if os.path.exists(CHECKPOINT_DIR):
        model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, f"best_model_{training_type}_val_acc.pth"), map_location=DEVICE))
        model_org.load_state_dict(torch.load(PRETRAINED_MODEL_PATH , map_location=DEVICE))
        model_org.eval()
        model.eval()
        patch_orig_all = []
        patch_all = []
        att1_all = []
        att2_all = []
        labels_all = []
        for bags, labels, ids in tqdm(test_loader, desc="Testing", leave=False):
            print(ids)
            if not bags:
                continue

            moved_bags = []
            valid_indices = []
            for i, bag in enumerate(bags):
                if bag.nelement() > 0:
                    moved_bags.append(bag.to(DEVICE))
                    valid_indices.append(i)

            if not moved_bags:
                continue

            labels = labels[valid_indices]
            labels_all.append(labels.cpu())

            if is_training:
                optimizer.zero_grad()

            # print("attention_tool", attention_tool)
            with torch.no_grad():
                outputs, _, patch_embeddings_orig, patch_embeddings_stacked_all, att_feature_orig_all, aggregated_feature2_all = model(moved_bags, model_org, attention_tool)
                # print(f"patch_embeddings_stacked_all shape: {patch_embeddings_stacked_all.shape}") # [16, 5248]
                # features_all.append(patch_embeddings_stacked_all)
                patch_orig_all.append(patch_embeddings_orig.cpu())  # Store original patch embeddings
                patch_all.append(patch_embeddings_stacked_all.cpu())
                att1_all.append(att_feature_orig_all.cpu())
                att2_all.append(aggregated_feature2_all.cpu())
                
            
        

        # Flatten all features
        patch_orig_all = torch.cat(patch_orig_all, dim=0).numpy()  # (N_bags, D1)
        print(f"patch_orig_all shape: {patch_orig_all.shape}")
        patch_all = torch.cat(patch_all, dim=0).numpy()            # (N_bags, D1)
        print(f"patch_all shape: {patch_all.shape}")
        att1_all = torch.cat(att1_all, dim=0).numpy()              # (N_bags, D2)
        print(f"att1_all shape: {att1_all.shape}")
        att2_all = torch.cat(att2_all, dim=0).numpy()              # (N_bags, D3)
        print(f"att2_all shape: {att2_all.shape}")
        labels_all = torch.cat(labels_all, dim=0).numpy()

        # Create figure with 2×2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))  # 2 rows, 2 columns

        # List of features and titles
        feature_sets = [
            (patch_orig_all, "t-SNE: Original Patch Embeddings"),
            (patch_all,      "t-SNE: Feature Patches (CAM-weighted)"),
            (att1_all,       "t-SNE: Aggregated Bag Features (Attention)"),
            (att2_all,       "t-SNE: Aggregated Bag Features (CAM-based)")
        ]

        # Plot each t-SNE projection
        for i, (features, title) in enumerate(feature_sets):
            tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
            features_2d = tsne.fit_transform(features)

            ax = axes[i // 2][i % 2]  # Convert i=0,1,2,3 to (0,0), (0,1), (1,0), (1,1)
            scatter = ax.scatter(
                features_2d[:, 0], features_2d[:, 1],
                c=labels_all, cmap='viridis', s=30, alpha=0.8
            )
            ax.set_title(title)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.grid(True)

        # Add shared legend
        fig.legend(*scatter.legend_elements(), title="KL Grade", loc='center right')

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 0.93, 1])  # Leave space for legend on the right

        # Save or show
        plt.savefig(os.path.join(IMG_SAVE_PATH, f"tsne_all_{training_type}_feedback{feedback_type}_4img.png"))
        # plt.show()