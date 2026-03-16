import numpy as np
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            confusion_matrix, roc_auc_score,
                            average_precision_score
                            )
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
import seaborn as sns
device = 'cuda'
import time
from datetime import datetime

from loss_st import (FocalLoss, 
                relation_distillation_loss, 
                KDLoss, LearnableContrastiveLoss,
                balanced_patch_supcon_loss
                )

Criterion = torch.nn.CrossEntropyLoss()  

alpha = torch.tensor([0.2, 0.8]).to(device)
patch_loss_criterion = FocalLoss(alpha=alpha, gamma=2.6).to(device)

loss_kd_feature = KDLoss(temperature=4.0)
lambda_l1 = 0.000002 

def train(aloader, nloader, model, batch_size, optimizer, device, loss_patch_ccl):
    with torch.set_grad_enabled(True):
        model.train()

        #patch level
        correct_patch = 0
        total_patch = 0
        y_all_true_patch = []
        y_all_pre_patch = []

        ninput_rgb, ninput_ir, nimglabel, npatch_label, nimgid = next(nloader)
        ainput_rgb, ainput_ir, aimglabel, apatch_label, aimgid = next(aloader)

        input_rgb = torch.cat((ninput_rgb, ainput_rgb), 0).to(device)
        input_ir = torch.cat((ninput_ir, ainput_ir), 0).to(device)
        
        stpatch_logits, stimg_logits, st_features, st_feat_proj, patch_ft_proj, tea_logits, tea_features  =  model(input_rgb, input_ir, mode = 'distill')          #model(input_rgb, input_ir, mode = 'distill') #use mode as student when exp with no KD
          
        stpatch_logits = stpatch_logits.permute(0,3,1,2) #[B, 2, 7,7]
        #print("nlabel", nlabel.size())
        #print("alabel", alabel.size())

        nlabel = nimglabel[0:batch_size]
        alabel = aimglabel[0:batch_size]

        # Slice the lists
        npatchlabel = npatch_label[0:batch_size].to(device)  # [B, 7, 7]
        apatchlabel = apatch_label[0:batch_size].to(device)  # [B, 7, 7]
        
        #print("npatchlabel.shape:",npatchlabel.shape)       
           

        labels = torch.cat((nlabel, alabel), 0).to(device) #image-level labels
        patch_labels = torch.cat((npatchlabel, apatchlabel), dim=0).to(device) #patch gt
        patch_labels = (patch_labels == 2).long() #for binary patch class
        
        
        loss_cls = Criterion(stimg_logits, labels)
        
        loss_patch_response = patch_loss_criterion(stpatch_logits, patch_labels)   
        loss_relation = relation_distillation_loss(st_features, tea_features)
     

        # print(f"classifi image Loss: {loss_cls}")
        # print(f"classifi patch Loss: {loss_patch_response}")

        KD = loss_kd_feature(st_features, tea_features)
       

        # Step 1: Create mask for image-level labels
        image_labels = labels.view(-1)         # Ensure shape is [64]
        pos_image_mask = (image_labels == 1)         # [64] → Boolean mask

        # Step 2: Use this to filter 64 image patches (shape [64, 512, 7, 7])
        B = patch_labels.shape[0]
        patch_ft_proj_4d = patch_ft_proj.view(B, 7, 7, 512).permute(0, 3, 1, 2)  # [64, 512, 7, 7]

        patch_ft_proj_pos = patch_ft_proj_4d[pos_image_mask]      # [N_pos, 512, 7, 7]
        patch_labels_pos = patch_labels[pos_image_mask]           # [N_pos, 7, 7]

        # Step 3: Flatten patches and labels
        patch_ft_proj_flat = patch_ft_proj_pos.permute(0, 2, 3, 1).reshape(-1, 512)  # [N_pos*49, 512]
        patch_labels_flat = patch_labels_pos.reshape(-1, 1).long()                   # [N_pos*49, 1]

        # Step 4: Compute patch contrastive loss
        if patch_ft_proj_flat.shape[0] > 0:
            Patch_CCL = loss_patch_ccl(patch_ft_proj_flat, patch_labels_flat)
        else:
            Patch_CCL = torch.tensor(0.0, device=patch_ft_proj.device, requires_grad=True)          
        
        
        ##print(f"CCL Loss: {CCL}")
        # print(f"KD Loss: {KD}")
        # print(f"Patch_CCL: {Patch_CCL}")
        # print(f"loss_relation: {loss_relation}")

        patch_features = st_features.permute(0,2,3,1)  #[B, 768, 7, 7] --> [B,7,7,768]
        patch_features = F.normalize(patch_features, dim=3)
        patch_supconloss = balanced_patch_supcon_loss(patch_features, patch_labels, labels, temperature=0.08)

        #print(f"Patch SupCon loss: {patch_supconloss.item():.4f}")      
       
        

        # ---- PATCH-LEVEL ACCURACY ----              
        patch_pred = stpatch_logits.argmax(1)  #-> [B, 7, 7]
        total_patch += patch_labels.numel()
        correct_patch += (patch_pred == patch_labels).sum().item()
        
        accuracy_patch = 100.0 * correct_patch / total_patch
        #print('train: patch level accuracy::', accuracy_patch)

        
        # --- FIRE CLASS RECALL PENALTY ---
        y_true = patch_labels.view(-1)
        y_pred = patch_pred.view(-1)

        fire_class = 1
        
        y_true = patch_labels.view(-1)
        probs = F.softmax(stpatch_logits, dim=1)
        fire_probs = probs[:, 1, :, :]  # [B, 7, 7] - fire probabilities only
        fire_probs_flat = fire_probs.reshape(-1)  # [B*7*7] - flattened fire probabilities
        
        fire_mask = (y_true == fire_class).float()
        
        # Soft metrics
        soft_tp = (fire_probs_flat * fire_mask).sum()
        soft_fn = ((1.0 - fire_probs_flat) * fire_mask).sum()
        
        soft_recall = soft_tp / (soft_tp + soft_fn + 1e-6)
        # Penalty (False Negative Rate)
        fnr_penalty = 1.0 - soft_recall
        penalty_loss = fnr_penalty * 100.0

        # Compute L1 regularization
        l1_regularization = 0
        for param in model.parameters():
            l1_regularization += torch.sum(torch.abs(param))             
        
        
        cost = (
                    275 * KD +
                    25 * loss_cls +                       
                    100 * loss_patch_response +
                    740 * loss_relation +
                    20 * patch_supconloss +              
                    40 * Patch_CCL +                    
                    80 * penalty_loss +                 
                    lambda_l1 * l1_regularization
                )

               
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
                
        return cost.item() , accuracy_patch  


def test(dataloader, model, batch_size, device, output_flag=False):
    model.eval()

    y_true_img, y_pred_img, y_score_img = [], [], []
    y_true_patch, y_pred_patch, y_score_patch = [], [], []

    # Track which patches belong to images predicted as class 1
    y_true_patch_fire_img, y_pred_patch_fire_img, y_score_patch_fire_img = [], [], []

    with torch.no_grad():
        for img_rgb, img_label, patch_label, imgid in dataloader:
            img_rgb = img_rgb.to(device)
            img_label = img_label.to(device)
            
            patch_label = patch_label.to(device)
            patch_label = (patch_label == 2).long() #for binary patch class

            start_time = time.time()
            stpatch_logits, stimg_logits, _, _ , _ = model(img_rgb, mode='student')
            end_time = time.time()
            time_taken = end_time - start_time

            # Image-level prediction
            probs_img = F.softmax(stimg_logits, dim=1)
            pred_img = probs_img.argmax(dim=1)
            y_true_img.append(img_label.item())
            y_pred_img.append(pred_img.item())
            y_score_img.append(probs_img.squeeze().cpu().numpy())  # shape [num_classes]

            
            # Patch-level prediction
            stpatch_logits = stpatch_logits.permute(0, 3, 1, 2)  # [B, 3, 7, 7]
            #print('stpatch_logits shape:',stpatch_logits.shape) #torch.Size([1, 2, 7, 7])
            
            probs_patch = F.softmax(stpatch_logits, dim=1)  # [B, 2, 7, 7]
            pred_patch = probs_patch.argmax(dim=1)  # [B, 7, 7]

            imgid = int(imgid[0])
            
            y_true_patch.extend(patch_label.view(-1).cpu().numpy())
            y_pred_patch.extend(pred_patch.view(-1).cpu().numpy())
            patch_probs = probs_patch.permute(0, 2, 3, 1)  # shape: [B, 7, 7, 2]
            patch_probs_flat = patch_probs.reshape(-1, 2)  # shape: [B*7*7, 2]
            y_score_patch.extend(patch_probs_flat.cpu().numpy())  # Store both class probs

            # If image is predicted as class 1 (fire), add its patches to the filtered lists
            if pred_img.item() == 1:
                y_true_patch_fire_img.extend(patch_label.view(-1).cpu().numpy())
                y_pred_patch_fire_img.extend(pred_patch.view(-1).cpu().numpy())
                y_score_patch_fire_img.extend(patch_probs_flat.cpu().numpy())

    # Convert to numpy arrays
    y_true_img = np.array(y_true_img)
    y_pred_img = np.array(y_pred_img)
    y_score_img = np.array(y_score_img)

    y_true_patch = np.array(y_true_patch)
    y_pred_patch = np.array(y_pred_patch)
    y_score_patch = np.array(y_score_patch)

    # Convert filtered patch arrays
    y_true_patch_fire_img = np.array(y_true_patch_fire_img)
    y_pred_patch_fire_img = np.array(y_pred_patch_fire_img)
    y_score_patch_fire_img = np.array(y_score_patch_fire_img)

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

            # AUC-ROC and AUC-PRC (only if num_classes == 2 or probabilities are provided)
            try:
                if num_classes == 2:
                    auc_roc = roc_auc_score(y_true, y_score[:, 1])
                    auc_prc = average_precision_score(y_true, y_score[:, 1])

                                        
                    # Plot and save ROC curve for binary classification
                    fpr_roc, tpr_roc, _ = roc_curve(y_true, y_score[:, 1])
                    plt.figure(figsize=(8, 6))
                    plt.subplot(1, 2, 1)
                    plt.plot(fpr_roc, tpr_roc, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.4f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'{name} Level ROC Curve')
                    plt.legend(loc="lower right")
                    plt.grid(True, alpha=0.3)
                    
                    # Plot and save PR curve for binary classification
                    precision_pr, recall_pr, _ = precision_recall_curve(y_true, y_score[:, 1])
                    plt.subplot(1, 2, 2)
                    plt.plot(recall_pr, precision_pr, color='blue', lw=2, label=f'PR curve (AUC = {auc_prc:.4f})')
                    baseline = np.sum(y_true) / len(y_true)  # Proportion of positive class
                    plt.axhline(y=baseline, color='red', linestyle='--', label=f'Baseline ({baseline:.4f})')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.title(f'{name} Level Precision-Recall Curve')
                    plt.legend(loc="lower left")
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(f'OUTPUT/{name}_auc_curves.png', dpi=300, bbox_inches='tight')
                    plt.close()


                
                else:
                    auc_roc = roc_auc_score(y_true, y_score, multi_class='ovo', average='macro')
                    auc_prc = average_precision_score(y_true, y_score, average='macro')
                log_file.write(f"AUC-ROC: {auc_roc:.4f}\n")
                log_file.write(f"AUC-PRC: {auc_prc:.4f}\n")
                print(f"AUC-ROC: {auc_roc:.4f}")
                print(f"AUC-PRC: {auc_prc:.4f}")
            except ValueError:
                log_file.write("AUC-ROC/PRC could not be computed.\n")
                print("AUC-ROC/PRC could not be computed.")

            fire_class_index = 1
            tp_fire = cm[fire_class_index, fire_class_index]
            fn_fire = np.sum(cm[fire_class_index, :]) - tp_fire
            recall_fire = tp_fire / (tp_fire + fn_fire + 1e-6)  # +1e-6 for numerical stability

            log_file.write(f"Recall (Fire class): {recall_fire:.4f}\n")
            print(f"Recall (Fire class): {recall_fire:.4f}")      

        # Function to plot filtered patch metrics (only patches from fire-predicted images)
        def print_filtered_patch_metrics(name, y_true, y_pred, y_score, num_classes, log_file):
            if len(y_true) == 0:
                log_file.write(f"\n----- {name} Level Metrics (No samples) -----\n")
                print(f"\n----- {name} Level Metrics (No samples) -----")
                return
                
            log_file.write(f"\n----- {name} Level Metrics -----\n")
            log_file.write(f"Number of samples: {len(y_true)}\n")
            print(f"\n----- {name} Level Metrics -----")
            print(f"Number of samples: {len(y_true)}")

            acc = np.mean(y_true == y_pred) * 100
            f1 = f1_score(y_true, y_pred, average='macro')
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

            log_file.write(f"Accuracy: {acc:.2f}%\n")
            log_file.write(f"Precision (Macro): {precision:.4f}\n")
            log_file.write(f"Recall (Macro): {recall:.4f}\n")
            log_file.write(f"F1 Score (Macro): {f1:.4f}\n")

            print(f"Accuracy: {acc:.2f}%")
            print(f"Precision (Macro): {precision:.4f}")
            print(f"Recall (Macro): {recall:.4f}")
            print(f"F1 Score (Macro): {f1:.4f}")

            # AUC-PRC for filtered patches
            try:
                if num_classes == 2 and len(np.unique(y_true)) > 1:
                    auc_prc = average_precision_score(y_true, y_score[:, 1])
                    
                    # Plot PR curve for filtered patches
                    precision_pr, recall_pr, _ = precision_recall_curve(y_true, y_score[:, 1])
                    plt.figure(figsize=(8, 6))
                    plt.plot(recall_pr, precision_pr, color='green', lw=2, label=f'PR curve (AUC = {auc_prc:.4f})')
                    baseline = np.sum(y_true) / len(y_true)
                    plt.axhline(y=baseline, color='red', linestyle='--', label=f'Baseline ({baseline:.4f})')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.title(f'{name} Level Precision-Recall Curve')
                    plt.legend(loc="lower left")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f'OUTPUT/{name}_filtered_pr_curve.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    log_file.write(f"AUC-PRC: {auc_prc:.4f}\n")
                    print(f"AUC-PRC: {auc_prc:.4f}")
                else:
                    log_file.write("AUC-PRC could not be computed (insufficient class diversity).\n")
                    print("AUC-PRC could not be computed (insufficient class diversity).")
            except ValueError as e:
                log_file.write(f"AUC-PRC could not be computed: {str(e)}\n")
                print(f"AUC-PRC could not be computed: {str(e)}")    


        
        os.makedirs("OUTPUT", exist_ok=True)
        log_path = os.path.join("OUTPUT", "metrics_log.txt")
        with open(log_path, 'w') as log_file:
            print_metrics("Image", y_true_img, y_pred_img, y_score_img, num_classes=2, 
                          log_file=log_file, cm_name='out_cm_image.png')
            print_metrics("Patch", y_true_patch, y_pred_patch, y_score_patch, num_classes=2, 
                          log_file=log_file, cm_name='out_cm_patch.png')
            
            # NEW: Print metrics for patches from fire-predicted images only
            print_filtered_patch_metrics("Patch (Fire-predicted Images Only)", 
                                       y_true_patch_fire_img, y_pred_patch_fire_img, 
                                       y_score_patch_fire_img, num_classes=2, log_file=log_file)

        
        
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
