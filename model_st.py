import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import CLIPModel, CLIPProcessor
from torchvision.transforms import Normalize
from model_teacher import CLIP_teacher

import torch
import torchvision.ops
from torch import nn


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      bias=bias)

    def forward(self, x):
        
        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation)
        return x



class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        max_ = self.fc(self.max_pool(x))
        return self.sigmoid(avg + max_) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat)) * x

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.channel = ChannelAttention(in_channels, reduction)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


class StudentModel(nn.Module):
    def __init__(self, teacher_weights_path, num_classes=2, modx='resnet'):#, training=True
        super(StudentModel, self).__init__()
        
        self.clip_teacher = CLIP_teacher()        
        self.clip_teacher.load_state_dict(torch.load(teacher_weights_path))
        
        for param in self.clip_teacher.parameters():
            param.requires_grad = False
        self.clip_teacher.eval()

        # Load pretrained ResNet (student)
        if modx == 'squeezenet':
            self.modx = models.squeezenet1_1(pretrained=True)  #([1, 512, 13, 13])
            self.conv = nn.Sequential(
                                    nn.Conv2d(in_channels=512, out_channels=768, kernel_size=3, stride=2, padding=1),  
                                    nn.BatchNorm2d(768),
                                    #nn.LayerNorm(768),
                                    nn.ReLU()
                                                )# (1, 768, 7, 7)
            
        elif modx == 'densenet':
            self.modx = models.densenet121(pretrained=True)  ##[1, 1024, 7, 7] for 224
            self.conv = nn.Sequential(                                   
                                    nn.Conv2d(in_channels=1024, out_channels=768, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(768),
                                    #nn.LayerNorm(768),
                                    nn.ReLU()
                                    ) #[1,768,7,7]
            
        elif modx == 'resnet':
            self.resnet = models.resnet50(pretrained=True)  # [B, 2048, 7,7] feature map
            self.modx = nn.Sequential(*list(self.resnet.children())[:-2])
            self.conv = nn.Sequential( 
                                    nn.Conv2d(in_channels=2048, out_channels=768, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(768),
                                    #nn.LayerNorm(768),
                                    nn.ReLU()
                                    ) #[1, 768, 7, 7] #224
            
                
        elif modx == 'mobilenetv2':
            self.mobilenet = models.mobilenet_v2(pretrained=True)
            self.modx = self.mobilenet.features  # output: [B, 1280, 7, 7] for input 224x224
            self.conv = nn.Sequential(
                                    nn.Conv2d(in_channels=1280, out_channels=768, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(768),
                                    #nn.LayerNorm(768),
                                    nn.ReLU()
                                )  # Output: [B, 768, 7, 7]
            

        elif modx == 'clip':
            self.modx_clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32') #[1, 768,7,7]
            for param in self.modx_clip.vision_model.parameters():
                param.requires_grad = False
            

        elif modx == "effnet":
            self.effnet = models.efficientnet_b0(pretrained=True)
            self.modx = self.effnet.features ## [B, 1280, 7, 7] if input is 224x224
            self.conv = nn.Sequential(
                                    nn.Conv2d(in_channels=1280, out_channels=768, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(768),
                                    nn.ReLU()
                                )        
                     

        self.conv_project = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=512, kernel_size=1, stride=1, padding=0),  # (B, 512, 7, 7)
            nn.BatchNorm2d(512),
            #nn.LayerNorm(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 512, 1, 1)
            nn.Flatten()                   # (B, 512)
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

        # Student's MLP classifier
        self.img_classifier = nn.Sequential(          
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64), 
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, num_classes) # fire-free/ fire-impacted 
        )

        self.patch_classifier = nn.Sequential(
            nn.Linear(512, 256), #(768, 256)
            #nn.BatchNorm1d(256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, 64), 
            #nn.BatchNorm1d(64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.35),
            #nn.Linear(64, 3) #fire, smoke, bg
            nn.Linear(64, 2)
        )

        self.patch_proj_layer = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),            
            nn.ReLU(),
            )

    def forward(self, rgb, ir=None, mode='student'):
        if hasattr(self, 'modx_clip'): #if using clip
            normalize = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                  std=[0.26862954, 0.26130258, 0.27577711])
            rgb_norm = torch.stack([normalize(img) for img in rgb])  # [B, 3, 224, 224]
            
            # Pass to CLIP visual encoder
            outputs = self.modx_clip.vision_model(pixel_values=rgb_norm)
            patch_feats = outputs.last_hidden_state  # [B, 50, 768]
            
            # Remove CLS token and reshape to [B, 768, 7, 7]
            B, N, D = patch_feats.shape  # N=50 (1 CLS + 49 patches)
            feat_wo_cls = patch_feats[:, 1:, :]  # [B, 49, 768]
            st_feat = feat_wo_cls.permute(0, 2, 1).reshape(B, D, 7, 7)  # [B, 768, 7, 7]
        else:
            #non-clip backbone     # Student's forward pass         
            try:
                st_feat = self.modx.features(rgb)  
            except AttributeError:  # If the model doesn't have a `.features` attribute (like ResNet)
                st_feat = self.modx(rgb)  # For models like ResNet that don't have a .features attribute                     

            st_feat = self.conv(st_feat) # B, C, H, W =st_feat.shape #B, 768, 7, 7

        st_feat = self.deform_conv(st_feat)
        st_feat = self.cbam(st_feat)

        #print('st_feat shape:', st_feat.shape) #torch.Size([4, 768, 7, 7])

        #patch level classification
        B, C, H, W = st_feat.shape
        patch_ft = st_feat.permute(0, 2, 3, 1).reshape(B * H * W, C)  # Reshape patches
        patch_ft_proj =  self.patch_proj_layer(patch_ft)
        
        st_patch_ft = self.patch_classifier(patch_ft_proj)
        stpatch_logits = st_patch_ft.view(B, H, W, -1)  # Reshape to (B, H, W, num_classes)

        #image_level
        #st_feat similarity with text encoder [2, 512]
        st_feat_proj = self.conv_project(st_feat) #[B, 512]

        #image level classification
        stimg_logits = self.img_classifier(st_feat_proj)
        
        if mode == 'distill' and ir is not None:
            # Teacher's forward pass (CLIP)
            with torch.no_grad():
                teacher_logits, teacher_features = self.clip_teacher(ir) #teacher_features : B, N=50, D=768
                b, n, d = teacher_features.shape  
                h = w = int((n-1) ** 0.5)  # Assuming a square grid #7
                teacher_features = teacher_features[:, 1:, :].permute(0, 2, 1).reshape(b, d, h, w)
            return stpatch_logits, stimg_logits, st_feat, st_feat_proj, patch_ft_proj, teacher_logits, teacher_features
        
        else:
            return stpatch_logits, stimg_logits, st_feat, st_feat_proj, patch_ft_proj  #, teacher_logits, teacher_features

# Example usage
if __name__ == '__main__':
    model = StudentModel(teacher_weights_path="best_model_CLIP_teacher_IR2.pth", modx='densenet')
    rgb_input = torch.randn(4, 3, 224, 224).cuda()  # Batch of 4 RGB images
    ir_input = torch.randn(4, 3, 224, 224).cuda()  # Batch of 4 ir images
    model.cuda()

    studentpatch_logits, studentimg_logits, student_features, st_feat_proj, teacher_logits, teacher_features = model(rgb_input, ir_input)

    print("Student patch Logits:", studentpatch_logits.shape)
    print("Student image Logits:", studentimg_logits.shape)
    print("Teacher Logits:", teacher_logits.shape)
    print("Student Features:", student_features.shape)
    print("Teacher Features:", teacher_features.shape)
    print("student Feat projected:", st_feat_proj.shape)

# Student patch Logits: torch.Size([4, 7, 7, 3])
# Student image Logits: torch.Size([4, 2])
# Teacher Logits: torch.Size([4, 3])
# Student Features: torch.Size([4, 768, 7, 7])
# Teacher Features: torch.Size([4, 768, 7, 7])
# student Feat projected: torch.Size([4, 512])