# Cross-Modal Distillation for Real-time Wildfire Detection and Localization in Edge-Deployed Aerial Vehicles

A deep learning framework for fire detection using knowledge distillation from a CLIP-based teacher model to lightweight student models. The system performs both image-level and patch-level fire detection with high accuracy.

## Overview

This project implements a teacher-student knowledge distillation approach for fire detection:
- **Teacher Model**: CLIP-based vision transformer fine-tuned on infrared (IR) images
- **Student Models**: Lightweight CNN architectures (DenseNet, ResNet, MobileNetV2, etc.) trained on RGB images
- **Multi-level Detection**: Both image-level (fire-impacted vs fire-free) and patch-level (7×7 grid) classification

## Key Features

- 🔥 **Dual-level Fire Detection**: Image and patch-level predictions
- 🎓 **Knowledge Distillation**: Transfer learning from CLIP teacher to efficient student models
- 🧩 **Multiple Loss Functions**: 
  - Focal Loss for class imbalance
  - Contrastive Learning (SupCon, Learnable CCL)
  - Relation Distillation
  - Feature-level KD with Pearson Correlation
- 🏗️ **Modular Architecture**: Deformable convolutions and CBAM attention mechanisms
- 📊 **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PRC, FNR, FPR

## Project Structure

```
.
├── dataset_flame2/          # Training dataset
├── dataset_flame3/          # Additional test dataset
├── dataset_st.py            # Data loading and preprocessing
├── model_teacher.py         # CLIP-based teacher model
├── model_st.py              # Student model architectures
├── loss_st.py               # Custom loss functions
├── losses.py                # Supervised contrastive loss
├── train_st.py              # Training and evaluation functions
├── main_st.py               # Main training script
├── test_flame3.py           # Testing on dataset_flame3
├── test_patchhead.py        # Patch-level only testing
├── best_model_CLIP_teacher_IR2.pth  # Pre-trained teacher weights
├── bestmodel_densenet.pth   # Trained student model weights
└── patch_*.npy              # Pre-computed contrastive anchors
```

## Requirements

```bash
pip install torch torchvision transformers
pip install scikit-learn matplotlib seaborn
pip install Pillow numpy
pip install thop  # For FLOPs calculation
```

## Dataset Format

The dataset should be organized with:
- RGB and IR image pairs
- Image-level labels: `class_0` (fire-free), `class_1`/`class_2` (fire-impacted)
- Patch-level labels: JSON files with 7×7 grids (0: background, 1: smoke, 2: fire-flame ==> used in this project: 0/1 as 0: no-flame , 2 as 1: flame for FLAME2 dataset.)

```
dataset_flames2/
├── class_0/
├── class_1/
└── class_2/

RGB_train.txt  # List of RGB image paths
IR_train.txt   # List of IR image paths
patch_labels_train.json  # Patch-level annotations
```

## Training

### 1. Train Teacher Model (Optional)
The teacher model is already pre-trained. If you need to retrain:
```python
# Use CLIP teacher with IR images
# See model_teacher.py for architecture
```

### 2. Train Student Model
```bash
python main_st.py
```

**Configuration in `main_st.py`:**
```python
modx = 'densenet'  # Options: 'densenet', 'resnet', 'mobilenetv2', 'effnet'
batch_size = 32
max_epoch = 250
lr = 0.001
```

**Key Training Features:**
- Separate data loaders for normal and fire-impacted images
- Cycling through shorter dataset to match longer one
- Multi-loss optimization with weighted components
- Learning rate scheduling (StepLR)

## Testing

### Test on Standard Dataset
```bash
python test_flame3.py
```

### Test Patch-Level Performance Only
```bash
python test_patchhead.py
```

## Model Architecture

### Student Model Components

1. **Backbone**: DenseNet121 / ResNet50 / MobileNetV2 / EfficientNet-B0
2. **Feature Enhancement**:
   - Deformable Convolution (adaptive receptive fields)
   - CBAM Attention (channel + spatial attention)
3. **Classification Heads**:
   - Image-level: 2-class (fire-free/fire-impacted)
   - Patch-level: 2-class per patch (no-flame/flame)

### Loss Function Components

```python
Total Loss = 275×KD + 25×CE + 100×Focal + 740×Relation + 
             20×SupCon + 40×CCL + 80×FNR_Penalty + L1_Reg
```

- **KD**: Knowledge distillation (spatial PCC loss)
- **CE**: Cross-entropy for image classification
- **Focal**: Focal loss for patch classification (α=[0.2, 0.8], γ=2.6)
- **Relation**: Relation distillation between student-teacher features
- **SupCon**: Supervised contrastive loss for fire patches
- **CCL**: Learnable contrastive loss with multiple anchors
- **FNR_Penalty**: Soft recall penalty for fire class


### Patch-Level Visualization

The model outputs 7×7 patch predictions for fine-grained fire localization:
```
[0, 0, 0, 0, 0, 0, 0]
[0, 0, 1, 1, 0, 0, 0]
[0, 1, 1, 1, 1, 0, 0]
[0, 0, 1, 1, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0]
```

## Output Files

After testing, results are saved in `OUTPUT/` or `OUTPUT_flame3/`:
- `metrics_log.txt`: Detailed metrics
- `out_cm_image.png`: Image-level confusion matrix
- `out_cm_patch.png`: Patch-level confusion matrix
- `metrics_log_params.txt`: Model performance statistics

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{fire_detection_kd,
  title={Cross-Modal Distillation for Real-time Wildfire Detection and
	Localization in Edge-Deployed Aerial Vehicles},
  author={Medhavi Mishra et.al},
  year={2025},
  url={https://github.com/medh132/cmkd}
}
```

## License

MIT License

## Acknowledgments

- CLIP model from OpenAI
- SupConLoss implementation from [Yonglong Tian](https://github.com/HobbitLong/SupContrast)
- Deformable Convolution using torchvision.ops

## Contact

For questions or collaboration:
- Email: medhavi132@kaist.ac.kr
---

**Note**: Replace placeholder paths and credentials before deployment. Ensure pre-trained weights and anchor files are available.
