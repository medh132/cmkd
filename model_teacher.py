import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F
from torchvision import transforms
import os
import re


# prepare data for pytorch

# THREE CLASESS:
#     0:NN
#     1:YY
#     2:YN



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
print(f'use device: {DEVICE}')

class TransformerBlockWrapper(nn.Module):
    """Wraps a transformer block to return only the hidden states tensor (not a tuple)."""
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        # Extract only the hidden states from the block's output
        return self.block(hidden_states=x, attention_mask=None, causal_attention_mask=None)[0]


class CLIP_teacher(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        super(CLIP_teacher, self).__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)

        # Freeze all vision model parameters first
        for param in self.clip_model.vision_model.parameters():
            param.requires_grad = False

        # Wrap the last two blocks for training 
        self.last_two_blocks = nn.ModuleList([
            TransformerBlockWrapper(block)
            for block in self.clip_model.vision_model.encoder.layers[-2:]
        ])
        for block in self.last_two_blocks:
            for param in block.parameters():
                param.requires_grad = True

        # Define classifier on top of CLIP projection
        self.mlp1 = nn.Linear(512, 256)
        self.mlp2 = nn.Linear(256, 64)
        self.classifier = nn.Linear(64, 3)  # Example: 3 classes

    def forward(self, x):
        # Extract all hidden states from CLIP vision model
        vision_outputs = self.clip_model.vision_model(
            x,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = vision_outputs.hidden_states  # List of [B, N, D]

        # Get output from n-2 layer
        x = hidden_states[-3]  # shape: [batch, tokens, dim]

        # Pass through the last two transformer blocks
        for block in self.last_two_blocks:
            x = block(x)  # Already returns tensor due to wrapper
       
        x1 = x[:, 1:, :] #patch tokens
        x1 = x1.mean(dim=1) #global average pooling
        x1 = self.clip_model.visual_projection(x1)  # shape: [batch, 512]

        # Classifier head
        x1 = F.relu(self.mlp1(x1))
        x1 = F.relu(self.mlp2(x1))
        logits = self.classifier(x1)

        return logits, x  # Return both logits and token-level features
