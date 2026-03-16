import torch
import torch.nn as nn
from torchvision import models
from transformers import CLIPModel
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from thop import profile
import os
import numpy as np
from model_st import DeformableConv2d, CBAM
import json
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


import random

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    
    # For deterministic behavior 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed before creating model and dataloader
set_seed(22)



from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            confusion_matrix, roc_auc_score,
                            average_precision_score
                            )
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
import time
from datetime import datetime


# Configuration
#modx = 'resnet'
modx = 'densenet'
#modx = 'mobilenetv2'
#modx = 'effnet'
model_path = "bestmodel_densenet.pth"  # Path to your saved model
num_classes = 2
data_dir = "dataset_flames2" 

class StudentModelInference(nn.Module):
    def __init__(self, num_classes=2, modx='resnet'):
        super(StudentModelInference, self).__init__()
        
        # NOTE: No teacher model loaded for inference
        
        # Load pretrained backbone (student)
        if modx == 'squeezenet':
            self.modx = models.squeezenet1_1(pretrained=False)  # Set to False since we'll load weights
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=768, kernel_size=3, stride=2, padding=1),  
                nn.BatchNorm2d(768),
                nn.ReLU()
            )
            
        elif modx == 'densenet':
            self.modx = models.densenet121(pretrained=True)
            self.conv = nn.Sequential(                                   
                nn.Conv2d(in_channels=1024, out_channels=768, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(768),
                nn.ReLU()
            )
            
        elif modx == 'resnet':
            self.resnet = models.resnet50(pretrained=True)
            self.modx = nn.Sequential(*list(self.resnet.children())[:-2])
            self.conv = nn.Sequential( 
                nn.Conv2d(in_channels=2048, out_channels=768, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(768),
                nn.ReLU()
            )
            
        elif modx == 'mobilenetv2':
            self.mobilenet = models.mobilenet_v2(pretrained=True)
            self.modx = self.mobilenet.features
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=1280, out_channels=768, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(768),
                nn.ReLU()
            )

        elif modx == 'clip':
            self.modx_clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')

        elif modx == "effnet":
            self.effnet = models.efficientnet_b0(pretrained=True)
            self.modx = self.effnet.features
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=1280, out_channels=768, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(768),
                nn.ReLU()
            )

        self.conv_project = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        
        self.deform_conv = DeformableConv2d(
            in_channels=768,
            out_channels=768,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )       
          
              
        self.cbam = CBAM(768)
        

        # # Student's MLP classifier
        # self.img_classifier = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256), 
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(256, 64), 
        #     nn.ReLU(),
        #     nn.BatchNorm1d(64),
        #     nn.Linear(64, num_classes)
        # )

        self.patch_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, 64), 
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(64, 2)
        )

        self.patch_proj_layer = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

    def forward(self, rgb):
        if hasattr(self, 'modx_clip'):
            normalize = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                  std=[0.26862954, 0.26130258, 0.27577711])
            rgb_norm = torch.stack([normalize(img) for img in rgb])
            
            outputs = self.modx_clip.vision_model(pixel_values=rgb_norm)
            patch_feats = outputs.last_hidden_state
            
            B, N, D = patch_feats.shape
            feat_wo_cls = patch_feats[:, 1:, :]
            st_feat = feat_wo_cls.permute(0, 2, 1).reshape(B, D, 7, 7)
        else:
            try:
                st_feat = self.modx.features(rgb)  
            except AttributeError:
                st_feat = self.modx(rgb)                     

            st_feat = self.conv(st_feat)

        st_feat = self.deform_conv(st_feat)
        st_feat = self.cbam(st_feat)

        # Patch level classification
        B, C, H, W = st_feat.shape
        patch_ft = st_feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
        patch_ft_proj = self.patch_proj_layer(patch_ft)
        
        st_patch_ft = self.patch_classifier(patch_ft_proj)
        stpatch_logits = st_patch_ft.view(B, H, W, -1)

        # Image level
        #st_feat_proj = self.conv_project(st_feat)
        #stimg_logits = self.img_classifier(st_feat_proj)
        
        return stpatch_logits, st_feat,  patch_ft_proj #stimg_logits #st_feat_proj,


def load_student_weights_only(model, checkpoint_path):
    """
    Load only student model weights, excluding teacher weights
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = model.state_dict()

    # Filter out teacher-related weights
    student_state_dict = {}
    skipped_keys = []
    for key, value in checkpoint.items():
        if key.startswith('clip_teacher'):
            continue
        if key in model_state_dict and model_state_dict[key].shape == value.shape:
            student_state_dict[key] = value
        else:
            skipped_keys.append(key)
    
    model.load_state_dict(student_state_dict, strict=False)
    
    print(f"Loaded student weights from {checkpoint_path}")
    if skipped_keys:
        print(f"Skipped {len(skipped_keys)} keys (mismatch or not in model):")
        for k in skipped_keys:
            print(f"  - {k}")
    return model



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize student model for inference
model = StudentModelInference(num_classes=num_classes, modx=modx)

# Load only student weights
model = load_student_weights_only(model, model_path)
model.to(device)
model.eval()
  

def test(dataloader, model, device, output_flag=True):
    model.eval()

    y_true_img, y_pred_img, y_score_img = [], [], []
    y_true_patch, y_pred_patch, y_score_patch = [], [], []

    with torch.no_grad():
        for img_rgb, img_label, patch_label, imgid in dataloader:
            img_rgb = img_rgb.to(device)
            img_label = img_label.to(device)
            
            patch_label = patch_label.to(device)
            patch_label = (patch_label == 2).long() #for binary patch class

            start_time = time.time()
            stpatch_logits, _ , _ = model(img_rgb) #stimg_logits
            end_time = time.time()
            time_taken = end_time - start_time         
            
            # Patch-level prediction
            stpatch_logits = stpatch_logits.permute(0, 3, 1, 2)  # [B, 3, 7, 7]
            #print('stpatch_logits shape:',stpatch_logits.shape) #torch.Size([1, 2, 7, 7])
            
            probs_patch = F.softmax(stpatch_logits, dim=1)  # [B, 2, 7, 7]
            pred_patch = probs_patch.argmax(dim=1)  # [B, 7, 7]

            # # Image-level prediction
            # probs_img = F.softmax(stimg_logits, dim=1)
            # pred_img = probs_img.argmax(dim=1)

            # assign a default value first (e.g., background = 0)
            pred_img = torch.tensor([0], device=pred_patch.device)

            # --- Flip image-level pred if any patch is fire (class 1) ---
            if (pred_patch.view(-1) == 1).any():
                pred_img = torch.tensor([1], device=pred_patch.device)
            elif (pred_patch.view(-1) == 0).all():
                pred_img = torch.tensor([0], device=pred_patch.device)

            y_true_img.append(img_label.item())
            y_pred_img.append(pred_img.item())
            # y_score_img.append(probs_img.squeeze().cpu().numpy())  # shape [num_classes]

            #save patch predictions #imgid_gt.json #imgid_gt.npy 
            patch_pred_save = pred_patch.view(7,7).int().tolist() #since B=1
            imgid = int(imgid[0])
            

            y_true_patch.extend(patch_label.view(-1).cpu().numpy())
            y_pred_patch.extend(pred_patch.view(-1).cpu().numpy())
            # [B, 2, 7, 7] → [B, 7, 7, 2]
            patch_probs = probs_patch.permute(0, 2, 3, 1)  # shape: [B, 7, 7, 2]
            patch_probs_flat = patch_probs.reshape(-1, 2)  # shape: [B*7*7, 2]
            y_score_patch.extend(patch_probs_flat.cpu().numpy())  # Store both class probs

    # Convert to numpy arrays
    y_true_img = np.array(y_true_img)
    y_pred_img = np.array(y_pred_img)
    # y_score_img = np.array(y_score_img)

    y_true_patch = np.array(y_true_patch)
    y_pred_patch = np.array(y_pred_patch)
    y_score_patch = np.array(y_score_patch)

    # print('y_pred_img:',y_pred_img)
    print('y_pred_patch:',y_pred_patch)

    if output_flag:
        def print_metrics(name, y_true, y_pred, y_score, num_classes, log_file, cm_name):
            log_file.write(f"\n----- {name} Level Metrics -----\n")
            print(f"\n----- {name} Level Metrics -----")

            acc = np.mean(y_true == y_pred) * 100
            f1 = f1_score(y_true, y_pred, average='macro')
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            cm = confusion_matrix(y_true, y_pred)

            fnr = []
            fpr = []
            for i in range(num_classes):
                tp = cm[i, i]
                fn = np.sum(cm[i, :]) - tp
                fp = np.sum(cm[:, i]) - tp
                tn = np.sum(cm) - tp - fn - fp
                fnr_i = fn / (fn + tp + 1e-6)
                fpr_i = fp / (fp + tn + 1e-6)
                fnr.append(fnr_i)
                fpr.append(fpr_i)              

            log_file.write(f"Accuracy: {acc:.2f}%\n")
            log_file.write(f"Precision (Macro): {precision:.4f}\n")
            log_file.write(f"Recall (Macro): {recall:.4f}\n")
            log_file.write(f"F1 Score (Macro): {f1:.4f}\n")
            log_file.write(f"False Negative Rate per class: {fnr}\n")
            log_file.write(f"False Positive Rate per class: {fpr}\n")
            log_file.write(f"Confusion Matrix:\n{cm}\n")
            log_file.write("\nClassification Report:\n")
            log_file.write(classification_report(y_true, y_pred, digits=4))

            print(f"Accuracy: {acc:.2f}%")
            print(f"Precision (Macro): {precision:.4f}")
            print(f"Recall (Macro): {recall:.4f}")
            print(f"F1 Score (Macro): {f1:.4f}")
            print("False Negative Rate per class:", [round(x, 4) for x in fnr])
            print("False Positive Rate per class:", [round(x, 4) for x in fpr])
            print("Confusion Matrix:\n", cm)

            # Save confusion matrix heatmap
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
            plt.title(f'{name} Level Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(f'OUTPUT/{cm_name}')
            plt.close()

            ##AUC-ROC and AUC-PRC (only if num_classes == 2 or probabilities are provided)
            try:
                if num_classes == 2:
                    auc_roc = roc_auc_score(y_true, y_score[:, 1])
                    auc_prc = average_precision_score(y_true, y_score[:, 1])               
                
                else:
                    auc_roc = roc_auc_score(y_true, y_score, multi_class='ovo', average='macro')
                    auc_prc = average_precision_score(y_true, y_score, average='macro')
                # log_file.write(f"AUC-ROC: {auc_roc:.4f}\n")
                # log_file.write(f"AUC-PRC: {auc_prc:.4f}\n")
                # print(f"AUC-ROC: {auc_roc:.4f}")
                # print(f"AUC-PRC: {auc_prc:.4f}")
            except ValueError:
                log_file.write("AUC-ROC/PRC could not be computed.\n")
                print("AUC-ROC/PRC could not be computed.")

            fire_class_index = 1
            tp_fire = cm[fire_class_index, fire_class_index]
            fn_fire = np.sum(cm[fire_class_index, :]) - tp_fire
            recall_fire = tp_fire / (tp_fire + fn_fire + 1e-6)  # +1e-6 for numerical stability

            #log_file.write(f"Recall (Fire class): {recall_fire:.4f}\n")
            #print(f"Recall (Fire class): {recall_fire:.4f}")      
            
                    
        
        os.makedirs("OUTPUT", exist_ok=True)
        log_path = os.path.join("OUTPUT", "flame2metrics_log.txt")
        with open(log_path, 'w') as log_file:
            # print_metrics("Image", y_true_img, y_pred_img, y_score_img, num_classes=2, 
            #               log_file=log_file, cm_name='out_cm_image.png')
            print_metrics("Patch", y_true_patch, y_pred_patch, y_score_patch, num_classes=2, 
                          log_file=log_file, cm_name='out_cm_patch.png')
        
        
    # Return image-level and patch-level accuracy
    img_acc = np.mean(y_true_img == y_pred_img) * 100
    patch_acc = np.mean(y_true_patch == y_pred_patch) * 100


    cm = confusion_matrix(y_true_patch, y_pred_patch)
    fire_class_index = 1
    tp_fire = cm[fire_class_index, fire_class_index]
    fn_fire = np.sum(cm[fire_class_index, :]) - tp_fire
    recall_fire = (tp_fire / (tp_fire + fn_fire + 1e-6))*100  # +1e-6 for numerical stability
    print('val: patch level recall_fire:', recall_fire)

    
    return img_acc, patch_acc , recall_fire, time_taken



if __name__ == "__main__":
    import random  

    class CustomImageDataset(Dataset):
        def __init__(self, data_dir, patch_label_json, is_normal=True):
            self.data_dir = data_dir
            # self.mode = mode

            # Load patch labels from JSON
            with open(patch_label_json, 'r') as f:
                self.patch_label_dict = json.load(f)

            self.rgb_paths_all, self.labels_all = self.load_data()

            
            self.rgb_paths, self.labels = [], []
            for rgb, label in zip(self.rgb_paths_all, self.labels_all):
                    self.rgb_paths.append(rgb)
                    self.labels.append(label)

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            

        def load_data(self):
            
            rgb_file = f'{self.data_dir}/RGB_test.txt' #f'RGB_{self.mode}.txt'
            
            rgb_paths, labels = [], []
                        
            with open(rgb_file, 'r') as file:
                for line in file:
                    path = line.strip()
                    rgb_paths.append(path)
                    if "class_0" in path:
                        labels.append(0)
                    elif "class_1" in path or "class_2" in path:
                        labels.append(1)

            # Combine paths and labels to keep them aligned
            data = list(zip(rgb_paths, labels))

            # Randomly select 100 samples (without replacement)
            sampled_data = random.sample(data, 100)

            # Unzip back into separate lists
            rgb_paths, labels = zip(*sampled_data)
            rgb_paths, labels = list(rgb_paths), list(labels)
                
                    
            return rgb_paths, labels

        def __len__(self):
            return len(self.rgb_paths)

        def __getitem__(self, idx):
            img_rgb = Image.open(self.rgb_paths[idx]).convert("RGB")
            

            img_rgb = self.transform(img_rgb)
            

            img_label = self.labels[idx]

            # Extract image ID (assumes filename without extension is the ID)
            img_id = os.path.splitext(os.path.basename(self.rgb_paths[idx]))[0]
           
            # Retrieve patch label from JSON
            patch_label = self.patch_label_dict.get(img_id)
            if patch_label is None:
                raise ValueError(f"Patch label for image ID {img_id} not found in JSON.")
            
            patch_label = torch.tensor(patch_label, dtype=torch.long)  # Ensures shape [7, 7]

            
            return img_rgb, img_label, patch_label, img_id  # No IR in test mode
    
   
    test_loader = DataLoader(
                    CustomImageDataset(data_dir=data_dir, patch_label_json= f'{data_dir}/patch_labels_test.json'),
                    batch_size=1,
                    shuffle=False,
                    generator=torch.Generator(device='cpu'),
                    num_workers=0,
                    pin_memory=False,
                    )
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        print("Please ensure the model file is in the current directory.")
    
       
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
       
    accuracy_img, accuracy_patch, recall_fire, time = test(test_loader, model, device, output_flag=True)
        
    print(f"validation: patch accuracy {accuracy_patch:.4f}%, img accuracy {accuracy_img:.4f}%")   
        
              



def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def calculate_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def calculate_gflops(model, input_shape=(1, 3, 224, 224), device='cpu'):
    """Calculate GFLOPs using thop library"""
    try:
        model.eval()
        dummy_input = torch.randn(input_shape).to(device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        gflops = flops / 1e9
        print(f"GFLOPs: {gflops}, Parameters: {params}")
        return gflops
    except Exception as e:
        print(f"Error calculating GFLOPs: {e}")
        return None


def get_model_metrics(model_path, modx='resnet', num_classes=2, input_shape=(1, 3, 224, 224)):
    """
    Get comprehensive metrics for student model only
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize student model for inference
    model = StudentModelInference(num_classes=num_classes, modx=modx)
    
    # Load only student weights
    model = load_student_weights_only(model, model_path)
    model.to(device)
    model.eval()

    total_params = 0  # <-- Fix: Initialize this first
    print("Trainable parameters:\n")
    for name, param in model.named_parameters():
        if param.requires_grad:
            #print(f"{name}: {param.numel()}")
            total_params += param.numel()

    print(f"\nTotal trainable parameters: {total_params}")
    
    # Calculate metrics
    total_params, trainable_params = count_parameters(model)
    model_size_mb = calculate_model_size(model)
    gflops = calculate_gflops(model, input_shape, device)
    
    # Print results
    print("="*60)
    print("STUDENT MODEL METRICS (Inference Mode)")
    print("="*60)
    print(f"Model Architecture: {modx}")
    print(f"Number of Classes: {num_classes}")
    print(f"Input Shape: {input_shape}")
    print("-"*60)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: {model_size_mb:.2f} MB")
    if gflops is not None:
        print(f"GFLOPs: {gflops:.3f}")
    print("="*60)
    
    return {
        'model': model,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb,
        'gflops': gflops
    }


MODEL_PATH = "bestmodel_densenet.pth"  # Path to your saved model
#MODEL_PATH = f"best_model_student_{modx}.pth"
#MODX = 'densenet'  # Model architecture
NUM_CLASSES = 2
INPUT_SHAPE = (1, 3, 224, 224) 


if not os.path.exists(MODEL_PATH):
    print(f"Model file {MODEL_PATH} not found!")
    print("Please ensure the model file is in the current directory.")
else:
    # Get model metrics
    results = get_model_metrics(
        model_path=MODEL_PATH,
        modx=modx,
        num_classes=NUM_CLASSES,
        input_shape=INPUT_SHAPE
    )


summary_data = {
    'model_architecture': modx,
    'num_classes': NUM_CLASSES,
    'input_shape': INPUT_SHAPE,
    'total_parameters': results['total_params'],
    'trainable_parameters': results['trainable_params'],
    'model_size_mb': results['model_size_mb'],
    'gflops': results['gflops']
}

log_path = os.path.join("OUTPUT", "patchhead_metrics_log_params.txt")
with open(log_path, 'a') as f:  # 'a' = append mode
    f.write("====== Summary Data ======\n")
    f.write(json.dumps(summary_data, indent=4))
    f.write("\n\n")
  
print('Results saved to OUTPUT') 
