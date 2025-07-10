import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models # For using pre-trained backbones
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class PatchFeatureExtractor(nn.Module):
    def __init__(self, output_embedding_dim=128, num_channels=32, dropout_rate=0.4, adaptive_pool_output_size=(2, 2)):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels), # ADDED
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels), # ADDED
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels), # ADDED
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16x16 -> 8x8
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels*2), # ADDED
            nn.ReLU(),
            nn.Conv2d(num_channels*2, num_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels*2), # ADDED
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 8x8 -> 4x4
        )
        self.conv_block3 = nn.Sequential( # ADDED
            nn.Conv2d(num_channels*2, num_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels * 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # 4x4 -> 2x2
        )

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        fc_input_size = (num_channels * 4) * 2 * 2 # Recalculate based on 2x2 output
        self.fc = nn.Linear(fc_input_size, output_embedding_dim)

    def forward(self, x_patch): # x_patch is (C, H, W)
        x = self.conv_block1(x_patch)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        embedding = self.fc(x)
        embedding = self.dropout(embedding)
        return embedding


class MILAggregator(nn.Module):
    def __init__(self, input_embedding_dim, num_classes, aggregation_type='attention', dropout_rate=0.4):
        super().__init__()
        self.input_embedding_dim = input_embedding_dim
        self.aggregation_type = aggregation_type
        self.num_classes = num_classes

        if self.aggregation_type == 'attention':
            # Attention mechanism (a simple version)
            self.attention_V = nn.Linear(input_embedding_dim, 128)
            self.attention_U = nn.Linear(input_embedding_dim, 128)
            self.attention_w = nn.Linear(128, 1) # Outputs one score per patch embedding
        elif self.aggregation_type not in ['mean', 'max']:
            raise ValueError("Unsupported aggregation type")

        self.dropout = nn.Dropout(p=dropout_rate) # ADDED
        self.classifier = nn.Linear(input_embedding_dim, num_classes)

    def forward(self, patch_embeddings_bag): # patch_embeddings_bag is (N_patches, embedding_dim)
        if patch_embeddings_bag.nelement() == 0: # Handle empty bag
            # Return zeros or some other placeholder prediction
            # This implies the bag itself might be problematic
            print("Warning: MILAggregator received an empty patch_embeddings_bag.")
            return torch.zeros((1, self.num_classes), device=patch_embeddings_bag.device)


        if self.aggregation_type == 'mean':
            aggregated_features = torch.mean(patch_embeddings_bag, dim=0) # (embedding_dim)
        elif self.aggregation_type == 'max':
            aggregated_features, _ = torch.max(patch_embeddings_bag, dim=0) # (embedding_dim)
        elif self.aggregation_type == 'attention':
            # A = H (N_patches, embedding_dim)
            A_V = torch.tanh(self.attention_V(patch_embeddings_bag)) # (N_patches, 128)
            A_U = torch.sigmoid(self.attention_U(patch_embeddings_bag)) # (N_patches, 128) # Gating
            att_scores_unnorm = self.attention_w(A_V * A_U) # (N_patches, 1) element-wise product
            att_scores_unnorm = self.dropout(att_scores_unnorm) # Option 2: Dropout scores before softmax
            att_scores = F.softmax(att_scores_unnorm, dim=0) # (N_patches, 1)

            # Weighted sum of patch embeddings
            aggregated_features = torch.sum(att_scores * patch_embeddings_bag, dim=0) # (embedding_dim)
        else: # Should not happen due to init check
            raise ValueError("Unsupported aggregation type")

        # Add batch dimension if it was squeezed by mean/max/sum over dim=0
        if aggregated_features.ndim == 1:
            aggregated_features = aggregated_features.unsqueeze(0) # (1, embedding_dim)

        aggregated_features = self.dropout(aggregated_features) # APPLY DROPOUT
        logits = self.classifier(aggregated_features) # (1, num_classes)
        return logits, att_scores # ADDED


class CompleteMILModel(nn.Module):
    def __init__(self, feature_extractor_out_dim, num_classes, aggregation_type='attention'):
        super().__init__()
        self.patch_feature_extractor = PatchFeatureExtractor(output_embedding_dim=feature_extractor_out_dim)
        self.aggregator = MILAggregator(input_embedding_dim=feature_extractor_out_dim,
                                        num_classes=num_classes,
                                        aggregation_type=aggregation_type)

    def forward(self, list_of_patch_bags): # list_of_patch_bags: list of tensors, each (N_i, C, H, W)
        batch_logits = []
        batch_att_scores = []
        for patch_bag_tensor in list_of_patch_bags: # Iterate through samples in the batch
            # patch_bag_tensor is (N_i, C, H, W) for the i-th sample in batch
            if patch_bag_tensor.nelement() == 0 or patch_bag_tensor.shape[0] == 0:
                print("Warning: CompleteMILModel encountered an empty patch_bag_tensor for a sample.")
                # Create dummy logits for this problematic sample
                # This ensures the batch processing continues, but this sample won't contribute meaningfully
                dummy_logit = torch.zeros((1, self.aggregator.num_classes), device=next(self.parameters()).device)
                batch_logits.append(dummy_logit)
                continue

            # Process each patch in the bag
            # Need to pass patches one by one or as a batch to feature extractor
            # If feature_extractor takes (B, C, H, W), then patch_bag_tensor is already that.
            patch_embeddings_stacked = self.patch_feature_extractor(patch_bag_tensor) # (N_i, embedding_dim)

            # Aggregate and classify
            sample_logits, att_scores = self.aggregator(patch_embeddings_stacked) # (1, num_classes)
            batch_logits.append(sample_logits)
            batch_att_scores.append(att_scores)

        if not batch_logits: # If entire batch was problematic
            print("Warning: Entire batch resulted in no logits.")
            # This is a more severe issue, might need to return None or handle upstream
            # For now, return dummy based on expected batch size (though hard to know here)
            return torch.empty(0, self.aggregator.num_classes, device=next(self.parameters()).device)


        final_batch_logits = torch.cat(batch_logits, dim=0) # (batch_size, num_classes)
        final_batch_att_scores = torch.cat(batch_att_scores, dim=0) # (batch_size, num_patches)
        return final_batch_logits, final_batch_att_scores
    


from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, LayerCAM
    
class CompleteMILCamModel(nn.Module):
    def __init__(self, feature_extractor_out_dim, num_classes, aggregation_type='attention'):
        super().__init__()
        self.patch_feature_extractor = PatchFeatureExtractor(output_embedding_dim=feature_extractor_out_dim)
        self.aggregator = MILAggregator(input_embedding_dim=feature_extractor_out_dim,
                                        num_classes=num_classes,
                                        aggregation_type=aggregation_type)
        self.aggregator2 = MILAggregator(input_embedding_dim=feature_extractor_out_dim,
                                        num_classes=num_classes,
                                        aggregation_type=aggregation_type)
        
    def forward(self, list_of_patch_bags, model_org, gradcam_type): # list_of_patch_bags: list of tensors, each (N_i, C, H, W)
        batch_logits = []
        batch_att_scores = []
        batch_logits2 = []
        batch_att_scores2 = []
        attention_tool = None
        target_layer = [model_org.patch_feature_extractor.conv_block3[0]]
        

        
        for patch_bag_tensor in list_of_patch_bags: # Iterate through samples in the batch
            # patch_bag_tensor is (N_i, C, H, W) for the i-th sample in batch
            if patch_bag_tensor.nelement() == 0 or patch_bag_tensor.shape[0] == 0:
                print("Warning: CompleteMILModel encountered an empty patch_bag_tensor for a sample.")
                # Create dummy logits for this problematic sample
                # This ensures the batch processing continues, but this sample won't contribute meaningfully
                dummy_logit = torch.zeros((1, self.aggregator.num_classes), device=next(self.parameters()).device)
                batch_logits.append(dummy_logit)
                continue

            
            
            # Process each patch in the bag
            # Need to pass patches one by one or as a batch to feature extractor
            # If feature_extractor takes (B, C, H, W), then patch_bag_tensor is already that.
            patch_embeddings_stacked = self.patch_feature_extractor(patch_bag_tensor) # (N_i, embedding_dim)
            
            # Aggregate and classify
            sample_logits, att_scores = self.aggregator(patch_embeddings_stacked) # (1, num_classes)
            batch_logits.append(sample_logits)
            batch_att_scores.append(att_scores)

            # print("patch_bag_tensor shape", patch_bag_tensor.shape) # torch.Size([41, 1, 16, 16])
            # print("patch_embeddings_stacked shape", patch_embeddings_stacked.shape) #  torch.Size([41, 1, 16, 16])

            # Generate attention map
            if not gradcam_type == "original":
                target_class = sample_logits.argmax(dim=1).item()
                targets = [ ClassifierOutputTarget(target_class) ] * patch_bag_tensor.shape[0]
                
                # Override the outer no_grad here!
                with torch.set_grad_enabled(True):
                    logits, att_scores = model_org([patch_bag_tensor])
                    score = logits[0, target_class]
                    model_org.zero_grad()
                    score.backward(retain_graph=True)

                    # initialize here to prevend the layer hook!
                    if attention_tool == None:
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
                        elif gradcam_type == "ScoreCAM":
                            attention_tool = ScoreCAM(
                            model=model_org.patch_feature_extractor,   
                            target_layers=target_layer,
                            )
                        elif gradcam_type == "AblationCAM":
                            attention_tool = AblationCAM(
                            mmodel=model_org.patch_feature_extractor,       
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

                    attentionmap = attention_tool(
                            input_tensor=patch_bag_tensor,   # shape [41,1,16,16]
                            targets=targets
                        )
                # print("attentionmap shape: ", attentionmap.shape) # (41, 16, 16)
                attentionmap = torch.tensor(attentionmap, device=patch_bag_tensor.device)
                attentionmap_expanded = attentionmap.unsqueeze(1)  # (41, 1, 16, 16)
                # print("attentionmap_expanded shape: ", attentionmap_expanded.shape)
                # Combine the heatmap with patch
                patch_dot_attentionmap = torch.matmul(patch_bag_tensor, attentionmap_expanded)
                patch_dot_attentionmap_stacked = self.patch_feature_extractor(patch_dot_attentionmap)
                sample_logits2, att_scores2 = self.aggregator2(patch_dot_attentionmap_stacked) # (1, num_classes)
                batch_logits2.append(sample_logits2)
                batch_att_scores2.append(att_scores2)
                del patch_embeddings_stacked, patch_dot_attentionmap_stacked
                torch.cuda.empty_cache() 

        if not batch_logits: # If entire batch was problematic
            print("Warning: Entire batch resulted in no logits.")
            # This is a more severe issue, might need to return None or handle upstream
            # For now, return dummy based on expected batch size (though hard to know here)
            return torch.empty(0, self.aggregator.num_classes, device=next(self.parameters()).device)

        if not attention_tool == None:
            final_batch_logits = torch.cat(batch_logits2, dim=0) # (batch_size, num_classes)
            final_batch_att_scores = torch.cat(batch_att_scores2, dim=0) # (batch_size, num_patches)
            

        final_batch_logits = torch.cat(batch_logits, dim=0) # (batch_size, num_classes)
        final_batch_att_scores = torch.cat(batch_att_scores, dim=0) # (batch_size, num_patches)
        
        return final_batch_logits, final_batch_att_scores