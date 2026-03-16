from losses import SupConLoss
import torch.nn.functional as F
import torch.nn as nn
import torch

   
class PatchContrastiveLoss(nn.Module):
    def __init__(self, text_anchors, temperature=1.0):
        super(PatchContrastiveLoss, self).__init__()
        self.text_anchors = F.normalize(text_anchors, dim=1, p=2)  # [3, D]
        self.temperature = temperature

    def forward(self, patch_features, patch_labels):
        """
        patch_features: [N, D] - patch embeddings (e.g., 768 → projected to 512)
        patch_labels: [N] - int labels for patches (0 = bg, 1 = smoke, 2 = fire)
        """
        patch_features = F.normalize(patch_features, dim=1, p=2)  # [N, D]
        
        # Similarity between each patch and each text anchor
        logits = torch.matmul(patch_features, self.text_anchors.T)  # [N, 3]
        B = logits.shape[0] // (7 * 7)
        logits = logits / self.temperature
        logits = logits.view(B, 7, 7, 3).permute(0, 3, 1, 2)  # [B, 3, 7, 7]

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, patch_labels)
        return loss

    
def PCCloss(student_features: torch.Tensor,
            teacher_features: torch.Tensor,
            temperature: float = 4.0,
            epsilon: float = 1e-8) -> torch.Tensor:
    """
    Computes the Pearson Correlation Coefficient loss between student and teacher features.
    Args:
        student_features (torch.Tensor): Student feature map [B, C, H, W].
        teacher_features (torch.Tensor): Teacher feature map [B, C, H, W].
        temperature (float): Temperature for scaling.
        epsilon (float): Small value for numerical stability.
    Returns:
        torch.Tensor: A scalar loss value (1 - average PCC).
    """
    assert student_features.shape == teacher_features.shape, \
        "Shape mismatch between student and teacher features."

    # Flatten
    student_flat = student_features.reshape(student_features.size(0), -1) / temperature
    teacher_flat = teacher_features.reshape(teacher_features.size(0), -1) / temperature

    # Center
    student_centered = student_flat - student_flat.mean(dim=1, keepdim=True)
    teacher_centered = teacher_flat - teacher_flat.mean(dim=1, keepdim=True)

    # PCC
    numerator = (student_centered * teacher_centered).sum(dim=1)
    denominator = student_centered.norm(dim=1) * teacher_centered.norm(dim=1)

    corr = numerator / (denominator + epsilon)

    # Final loss
    loss = 1 - corr.mean()
    return loss


def spatial_PCCloss(student_features: torch.Tensor,
                    teacher_features: torch.Tensor,
                    temperature: float = 1.0,
                    epsilon: float = 1e-8) -> torch.Tensor:
    """
    Computes spatial Pearson Correlation Coefficient loss between student and teacher features.
    PCC is computed across channels per spatial location.
    
    Args:
        student_features (torch.Tensor): [B, C, H, W]
        teacher_features (torch.Tensor): [B, C, H, W]
        temperature (float): Optional temperature scaling.
        epsilon (float): Small value for numerical stability.
        
    Returns:
        torch.Tensor: A scalar loss (1 - average PCC over all spatial locations and batch).
    """
    assert student_features.shape == teacher_features.shape, \
        "Shape mismatch between student and teacher features."

    # Apply temperature scaling
    student = student_features / temperature
    teacher = teacher_features / temperature

    # Get mean over channel dimension: [B, 1, H, W]
    student_mean = student.mean(dim=1, keepdim=True)
    teacher_mean = teacher.mean(dim=1, keepdim=True)

    # Centered features: [B, C, H, W]
    student_centered = student - student_mean
    teacher_centered = teacher - teacher_mean

    # Compute numerator and denominator
    numerator = (student_centered * teacher_centered).sum(dim=1)  # [B, H, W]
    student_norm = student_centered.norm(dim=1)  # [B, H, W]
    teacher_norm = teacher_centered.norm(dim=1)  # [B, H, W]
    denominator = student_norm * teacher_norm + epsilon

    # PCC per spatial location: [B, H, W]
    pcc = numerator / denominator

    # Final loss: 1 - average PCC over all pixels and batch
    loss = 1 - pcc.mean()
    return loss


class KDLoss(nn.Module):
    def __init__(self, temperature: float = 3.0): #4.0
        super(KDLoss, self).__init__()
        self.temperature = temperature
        self.mse_loss = nn.MSELoss()
        self.epsilon = 1e-8

    def forward(self, student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
        # Ensure the input shapes are [B, 768, 7, 7]
        assert student_features.shape == teacher_features.shape, "Shape mismatch between student and teacher features."

        # Reshape the features to [B, -1] for MSE calculation
        # print('student features shape before:',student_features.shape)
        # print('teacher features shape before:',teacher_features.shape)
        
        student_features = student_features.reshape(student_features.size(0), -1)
        teacher_features = teacher_features.reshape(teacher_features.size(0), -1)

        # Apply temperature scaling
        student_features = student_features / self.temperature
        teacher_features = teacher_features / self.temperature
        # print('student features shape after:',student_features.shape) #torch.Size([4, 37632])
        # print('teacher features shape after:',teacher_features.shape)
        
        # Calculate MSE Loss
        #loss = self.mse_loss(student_features, teacher_features)
        loss = spatial_PCCloss(student_features, teacher_features, self.temperature, self.epsilon)

        return loss



def relation_distillation_loss(student_feat, teacher_feat, normalize=True):
    B, C, H, W = student_feat.shape
    N = H * W

    # Flatten
    st = student_feat.view(B, C, N)
    th = teacher_feat.view(B, C, N)

    # Optional normalization
    if normalize:
        st = st / (st.norm(dim=1, keepdim=True) + 1e-8)
        th = th / (th.norm(dim=1, keepdim=True) + 1e-8)

    # Affinity: [B, N, N]
    A_s = torch.bmm(st.transpose(1, 2), st)
    A_t = torch.bmm(th.transpose(1, 2), th)

    # Match relations
    loss = F.mse_loss(A_s, A_t)
    return loss




class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        alpha: Tensor of shape [num_classes], class weights (optional)
        gamma: focusing parameter
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] or [B, C] — raw predictions
        targets: [B, H, W] or [B] — ground truth labels
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)

        # Get probability of true class
        pt = torch.exp(-ce_loss)

        # Apply focal loss scaling
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def patch_supcon_loss(patch_features, patch_labels, image_labels, temperature=0.1):
    """
    Args:
        patch_features: Tensor [B, 7, 7, D]
        patch_labels: Tensor [B, 7, 7] (0=no-flame, 1=flame)
        image_labels: Tensor [B] (0=fire-free, 1=fire-impacted)
        temperature: float

    Returns:
        supcon_loss: scalar tensor
    """

    B, H, W, D = patch_features.shape
    device = patch_features.device

    # Select fire-impacted images only (label = 1)
    mask = image_labels == 1  # [B]
    if mask.sum() == 0:
        print("No fire-impacted images in batch.")
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Select relevant data
    selected_patch_feats = patch_features[mask]  # [B_fire, 7, 7, D]
    selected_patch_labels = patch_labels[mask]   # [B_fire, 7, 7]

    # Flatten patches
    feats = selected_patch_feats.reshape(-1, D)        # [B_fire*49, D]
    labels = selected_patch_labels.reshape(-1)         # [B_fire*49]

    # Normalize features
    feats = F.normalize(feats, dim=1)                  # [N, D]
    # Reshape to [N, 1, D] for SupConLoss
    feats = feats.unsqueeze(1) 


    # Use SupConLoss from timm
    supcon = SupConLoss(temperature=temperature)
    loss = supcon(feats, labels)

    return loss


def balanced_patch_supcon_loss(patch_features, patch_labels, image_labels, temperature=0.1):
    """
    Args:
        patch_features: Tensor [B, 7, 7, D]
        patch_labels: Tensor [B, 7, 7] (0=no-flame, 1=flame)
        image_labels: Tensor [B] (0=fire-free, 1=fire-impacted)
        temperature: float

    Returns:
        supcon_loss: scalar tensor
    """

    B, H, W, D = patch_features.shape
    device = patch_features.device

    # Select fire-impacted images only (label = 1)
    mask = image_labels == 1  # [B]
    if mask.sum() == 0:
        print("No fire-impacted images in batch.")
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Select relevant data
    selected_patch_feats = patch_features[mask]  # [B_fire, 7, 7, D]
    selected_patch_labels = patch_labels[mask]   # [B_fire, 7, 7]

    # Flatten patches
    feats = selected_patch_feats.reshape(-1, D)        # [N, D]
    labels = selected_patch_labels.reshape(-1)         # [N]

    # Find flame and no-flame indices
    flame_idx = (labels == 1).nonzero(as_tuple=True)[0]
    noflame_idx = (labels == 0).nonzero(as_tuple=True)[0]

    if len(flame_idx) == 0 or len(noflame_idx) == 0:
        # If only one class is present, skip contrastive loss
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Balance sampling: choose min(flame, noflame)
    num_samples = min(len(flame_idx), len(noflame_idx))
    sampled_flame_idx = flame_idx[torch.randperm(len(flame_idx))[:num_samples]]
    sampled_noflame_idx = noflame_idx[torch.randperm(len(noflame_idx))[:num_samples]]

    # Concatenate sampled indices
    selected_idx = torch.cat([sampled_flame_idx, sampled_noflame_idx], dim=0)

    # Select features and labels
    balanced_feats = feats[selected_idx]   # [2 * num_samples, D]
    balanced_labels = labels[selected_idx] # [2 * num_samples]

    # Normalize features
    balanced_feats = F.normalize(balanced_feats, dim=1)
    balanced_feats = balanced_feats.unsqueeze(1)  # [N, 1, D]

    # Contrastive Loss
    supcon = SupConLoss(temperature=temperature)
    loss = supcon(balanced_feats, balanced_labels)

    return loss


def contrastive_loss(dot_product, labels, temperature):
    """
    Numerically stable contrastive loss using log_softmax

    Args:
        dot_product: [batch_size, num_anchors] - similarities
        labels: [batch_size, 1] - binary labels (0 or 1)
        temperature: float - temperature scaling

    Returns:
        sim_loss: scalar - average contrastive loss
    """
    batch_size, num_anchors = dot_product.shape

    # Temperature scaling
    scaled_logits = dot_product / temperature  # [B, K]

    # Log-softmax over anchors
    log_probs = F.log_softmax(scaled_logits, dim=1)  # [B, K]

    # Create positive mask
    if labels.dim() == 1:
        labels = labels.unsqueeze(1)  # [B] -> [B, 1]
    labels_expanded = labels.expand(-1, num_anchors).float()  # [B, K]
    
    # Only positive anchors contribute
    masked_log_probs = log_probs * labels_expanded  # [B, K]

    total_loss = -torch.sum(masked_log_probs)
    num_positives = torch.sum(labels_expanded)

    if num_positives > 0:
        sim_loss = total_loss / num_positives
    else:
        sim_loss = torch.tensor(0.0, device=dot_product.device, requires_grad=True)

    return sim_loss


class LearnableContrastiveLoss(nn.Module):
    def __init__(self, positive_patch_anchor, negative_patch_anchor, temperature=1.0, alpha=0.5, class_weights=None):
        """
        Contrastive Loss with learnable multi-anchors, initialized from static anchors

        Args:
            positive_patch_anchor: torch.Tensor [2, 512] - initial positive anchors
            negative_patch_anchor: torch.Tensor [2, 512] - initial negative anchors
            temperature: float
            alpha: float
            class_weights: [neg_weight, pos_weight]
        """
        super(LearnableContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

        # Register as learnable parameters
        self.positive_anchor = nn.Parameter(positive_patch_anchor.clone().detach())
        self.negative_anchor = nn.Parameter(negative_patch_anchor.clone().detach())

        #fixed anchors #also remove from optimizer
        # self.positive_anchor = positive_patch_anchor.clone().detach()
        # self.negative_anchor = negative_patch_anchor.clone().detach()
       

        if class_weights is None:
            self.class_weights = torch.tensor([1.0, 1.0])
        else:
            self.class_weights = class_weights

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, dim=1)
        self.class_weights = self.class_weights.to(device)

        pos_anchors = F.normalize(self.positive_anchor, dim=1)
        neg_anchors = F.normalize(self.negative_anchor, dim=1)

        pos_sim = torch.matmul(features, pos_anchors.t())  # [batch_size, 2]
        neg_sim = torch.matmul(features, neg_anchors.t())  # [batch_size, 2]

        PL = contrastive_loss(pos_sim, labels, self.temperature)
        NL = contrastive_loss(neg_sim, 1 - labels, self.temperature)

        weighted_PL = PL * self.class_weights[1]
        weighted_NL = NL * self.class_weights[0]
        total_loss = self.alpha * weighted_PL + (1 - self.alpha) * weighted_NL
        

        return total_loss

