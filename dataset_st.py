import torch.utils.data as data
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os 
import json

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, patch_label_json, mode='train', is_normal=True):
        self.data_dir = data_dir
        self.mode = mode

        # Load patch labels from JSON
        with open(patch_label_json, 'r') as f:
            self.patch_label_dict = json.load(f)

        self.rgb_paths_all, self.ir_paths_all, self.labels_all = self.load_data()

        if self.mode == 'train':
            # Filter based on normal (label == 0) or abnormal (label == 1)
            self.rgb_paths, self.ir_paths, self.labels = [], [], []
            for rgb, ir, label in zip(self.rgb_paths_all, self.ir_paths_all, self.labels_all):
                if (is_normal and label == 0) or (not is_normal and label == 1):
                    self.rgb_paths.append(rgb)
                    self.ir_paths.append(ir)
                    self.labels.append(label)
        else:
            self.rgb_paths, self.ir_paths, self.labels = [], [], []
            for rgb, ir, label in zip(self.rgb_paths_all, self.ir_paths_all, self.labels_all):
                    self.rgb_paths.append(rgb)
                    self.ir_paths.append(ir)
                    self.labels.append(label)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        

    def load_data(self):
        
        rgb_file = f'{self.data_dir}/RGB_{self.mode}.txt' #f'RGB_{self.mode}.txt'
        ir_file = f'{self.data_dir}/IR_{self.mode}.txt' #f'IR_{self.mode}.txt'

        rgb_paths, ir_paths, labels = [], [], []
                    
        with open(rgb_file, 'r') as file:
            for line in file:
                path = line.strip()
                rgb_paths.append(path)
                if "class_0" in path:
                    labels.append(0)
                elif "class_1" in path or "class_2" in path:
                    labels.append(1)

        with open(ir_file, 'r') as file:
            for line in file:
                path = line.strip()
                ir_paths.append(path)

        assert len(rgb_paths) == len(ir_paths) == len(labels), "Mismatch between RGB, IR paths, and labels."

        # Verify that RGB and IR paths are properly paired
        for i in range(len(rgb_paths)):
            #print(rgb_paths[i])
            
            rgb_label = 0 if "class_0" in rgb_paths[i] else 1
            ir_label = 0 if "class_0" in ir_paths[i] else 1
            rgb_img = os.path.basename(rgb_paths[i])
            #print('rgb_img name:',rgb_img)
            ir_img = os.path.basename(ir_paths[i])
            #print('ir_img name:',ir_img)
            assert rgb_label == ir_label, f"RGB-IR label mismatch at index {i}"
            assert rgb_img == ir_img, f"RGB-IR image name mismatch at index {i}: {rgb_img} != {ir_img}"
        
        return rgb_paths, ir_paths, labels

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        img_rgb = Image.open(self.rgb_paths[idx]).convert("RGB")
        img_ir = Image.open(self.ir_paths[idx]).convert("RGB")

        img_rgb = self.transform(img_rgb)
        img_ir = self.transform(img_ir)

        img_label = self.labels[idx]

        # Extract image ID 
        img_id = os.path.splitext(os.path.basename(self.rgb_paths[idx]))[0]
        
        # Retrieve patch label from JSON
        patch_label = self.patch_label_dict.get(img_id)
        if patch_label is None:
            raise ValueError(f"Patch label for image ID {img_id} not found in JSON.")
        
        patch_label = torch.tensor(patch_label, dtype=torch.long)  # Ensures shape [7, 7]

        if self.mode == 'train':
            return img_rgb, img_ir, img_label, patch_label, img_id
        else:
            return img_rgb, img_label, patch_label, img_id  # No IR in test mode
    

if __name__ == '__main__':
    
    data_dir = "dataset_flames2"
    batch_size = 2
    device = 'cpu'  

    train_nloader = DataLoader(
        CustomImageDataset(data_dir=data_dir, patch_label_json= f'{data_dir}/patch_labels_train.json'  ,mode='train', is_normal=True),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device=device),
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )

    train_aloader = DataLoader(
        CustomImageDataset(data_dir=data_dir, patch_label_json= f'{data_dir}/patch_labels_train.json', mode='train', is_normal=False),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device=device),
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )

    test_loader = DataLoader(
        CustomImageDataset(data_dir=data_dir, patch_label_json= f'{data_dir}/patch_labels_test.json', mode='test'),
        batch_size=1,
        shuffle=False,
        generator=torch.Generator(device=device),
        num_workers=0,
        pin_memory=False,
        )
   

    for rgb, ir, img_label, patch_label,imgid in train_nloader:
        print("Normal batch - RGB:", rgb.shape, "IR:", ir.shape, "img_Label:", len(img_label))
        print("patch_label shape:", len(patch_label), len(patch_label[0]), len(patch_label[0][0]))
        break

    for rgb, ir, img_label, patch_label, imgid in train_aloader:
        print("Abnormal batch - RGB:", rgb.shape, "IR:", ir.shape, "img_Label:", img_label.shape, "patch_label shape:", patch_label.shape)
        break
    
    for rgb, img_label, patch_label, imgid in test_loader:
        print("test - RGB:", rgb.shape, "img_Label:", img_label.shape, "patch_label:", patch_label.shape)
        break
