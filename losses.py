from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from enum import Enum

class LossType(Enum):
    ORDINAL = 1
    ORDINAL_AND_FOCAL = 2

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

        # target_matrix: (batch_size, K-1)\
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

    def coral_focal_loss(self, logits, targets, kl_logits): # kl_logits is for class weights
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

        if self.class_weights is not None:
            # sample_weights: [batch] -> [batch, 1] so it can broadcast along num_classes
            sample_weights = self.class_weights[kl_logits].view(-1, 1) 
            loss_matrix = loss_matrix * sample_weights  # broadcast automatically

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
            
            total_loss += l
            loss_dict[task] = l.item()

        total_loss = total_loss / len(outputs)  # average over tasks

        return total_loss, loss_dict

def coral_predict(logits):
    """
    logits: (batch_size, K-1)
    return: predicted class (batch_size,)
    """
    prob = torch.sigmoid(logits)   # (batch, K-1)
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
