
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
from model_st import StudentModel
from dataset_st import CustomImageDataset
from train_st import train, test
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
import re
from datetime import datetime
from itertools import cycle
from loss_st import LearnableContrastiveLoss
import numpy as np

# Set your parameters
data_dir = "dataset_flames2"
batch_size = 32 
device = 'cuda'  
max_epoch = 2 #250
lr = 0.001 

#modx = 'resnet'
modx = 'densenet'
#modx = 'mobilenetv2'
#modx = 'effnet'


train_nloader = DataLoader(
        CustomImageDataset(data_dir=data_dir,patch_label_json= f'{data_dir}/patch_labels_train.json', mode='train', is_normal=True),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device='cpu'),
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )

train_aloader = DataLoader(
    CustomImageDataset(data_dir=data_dir, patch_label_json= f'{data_dir}/patch_labels_train.json', mode='train', is_normal=False),
    batch_size=batch_size,
    shuffle=True,
    generator=torch.Generator(device='cpu'),
    num_workers=0,
    pin_memory=False,
    drop_last=True
)

test_loader = DataLoader(
    CustomImageDataset(data_dir=data_dir, patch_label_json= f'{data_dir}/patch_labels_test.json', mode='test'),
    batch_size=1,
    shuffle=False,
    generator=torch.Generator(device='cpu'),
    num_workers=0,
    pin_memory=False,
    )

model_save_path = f"best_model_student_{modx}.pth"


model = StudentModel(teacher_weights_path="best_model_CLIP_teacher_IR2.pth", modx= modx)

model = model.cuda()

positive_patch_anchor=torch.from_numpy(np.load("patch_positive_flame_anchors.npy")).to(device)
negative_patch_anchor=torch.from_numpy(np.load("patch_negative_noflame_anchors.npy")).to(device)

loss_patch_ccl = LearnableContrastiveLoss(positive_patch_anchor=positive_patch_anchor,
                                    negative_patch_anchor=negative_patch_anchor,
                                    temperature=0.08,
                                    alpha=0.8,
                                    class_weights=torch.tensor([1.0, 10.0])  # optional
                                    )

optimizer = torch.optim.AdamW(
                            list(model.parameters())+ list(loss_patch_ccl.parameters()) ,  # ✅ include CCL params
                            lr=lr,
                            weight_decay= 1e-4   
                            ) 

scheduler = StepLR(optimizer, step_size=10, gamma=0.95)

best_patch_acc = 0.0
for epoch in range(1, max_epoch + 1):
    model.train()

    
    loadern_iter = iter(train_nloader)
    loadera_iter = iter(train_aloader)

    len_n = len(train_nloader)
    len_a = len(train_aloader)

    if len_n < len_a:
        shorter_iter = cycle(loadern_iter)
        longer_iter = loadera_iter
        steps_per_epoch = len_a
    else:
        shorter_iter = cycle(loadera_iter)
        longer_iter = loadern_iter
        steps_per_epoch = len_n

    batch_pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}", dynamic_ncols=True)

    # Collect metrics
    batch_patch_accuracies = []
    batch_losses = []
    batch_scores = []

    for _ in batch_pbar:
        loss_val, accuracy_patch = train(longer_iter, shorter_iter, model, batch_size, optimizer, device, loss_patch_ccl=loss_patch_ccl) #longer_iter=abnormal, #shorter_iter=normal
        batch_pbar.set_postfix(loss=f"{loss_val:.4f}")
        batch_patch_accuracies.append(accuracy_patch)
        #batch_patch_accuracies.append(recall_fire)
        batch_losses.append(loss_val)
        
        
    # End of epoch statistics
    avg_patch_acc = sum(batch_patch_accuracies) / len(batch_patch_accuracies)
    print(f"End of Epoch {epoch}: Avg Patch accuracy = {avg_patch_acc:.2f}% | Avg Loss = {sum(batch_losses)/len(batch_losses):.4f}")
    
    
    if epoch % 1  == 0:
        accuracy_img, accuracy_patch, recall_fire, time = test(test_loader, model, batch_size, device, output_flag=False)
        print(f"validation: patch accuracy {accuracy_patch:.4f}%, img accuracy {accuracy_img:.4f}%")
        if accuracy_patch > best_patch_acc:
            best_patch_acc = accuracy_patch
            at_epoch = epoch
            torch.save(model.state_dict(), model_save_path)
            print(f"Val: New best average patch accuracy ({accuracy_patch:.2f}%) achieved. Model saved to {model_save_path}")
        if accuracy_patch > 80.0 and recall_fire > 80.0:
            torch.save(model.state_dict(), f'bestmodel_{modx}_{epoch}.pth')
            print(f"best val: obtained and saved")
        
              
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
print(f'best patch level accuracy achieved during validation: {best_patch_acc} at epoch {at_epoch}')


################ test with output flag #############

model.load_state_dict(torch.load(f'bestmodel_{modx}.pth'))
#model.load_state_dict(torch.load(f'best_model_student_{modx}.pth'))

accuracy_img, accuracy_patch, recall_fire, time = test(test_loader, model, batch_size, device, output_flag=True) 
print(f"Test img accuracy: {accuracy_img}")
print(f"Test patch accuracy: {accuracy_patch}")
print(f"inference time {time}s")

