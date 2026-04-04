# AISEHack-CODE-CREW 
# FloodNet: Multi-Band Satellite Flood Segmentation

Deep learning pipeline for high-precision flood detection using multi-spectral satellite imagery. Built with PyTorch and optimized for Kaggle competition environments.

---

## Overview

FloodNet is an end-to-end semantic segmentation system designed to identify flood-affected regions from satellite imagery.

The system leverages:
- Multi-band geospatial data  
- Deep convolutional neural networks  
- Advanced loss functions and augmentation strategies  

---

## Problem Statement

Traditional flood monitoring systems suffer from:
- Delayed response times  
- Low spatial resolution  
- Limited generalization across diverse terrains  

This project addresses these challenges using an AI-based segmentation approach for faster and scalable flood detection.

---

## Solution Architecture

### Model Design

- Architecture: U-Net  
- Encoder: EfficientNet-B4 (pretrained on ImageNet)  
- Input Channels: 6  
- Output: Binary segmentation mask  

---

### Loss Function

```
Total Loss = Dice Loss + Focal Loss
```

- Dice Loss handles class imbalance  
- Focal Loss focuses on hard-to-classify pixels  

---

## Dataset

- Source: Kaggle competition dataset  
- Format: Multi-band `.tif` imagery  
- Labels: Binary segmentation masks  

### Structure

```
data/
 ├── image/
 ├── label/
 ├── prediction/
 └── split/
```

---

## System Design

### Data Pipeline

1. Load `.tif` images using rasterio  
2. Apply per-image normalization  
3. Convert masks to binary  
4. Batch processing using PyTorch DataLoader  

---

### Training Flow

```
Input → Model → Loss → Backpropagation → Optimizer → Update Weights
```

---

## Installation

### Enable GPU (Kaggle)

- Open Notebook settings  
- Select GPU (Tesla T4 or P100 recommended)  

---

### Install Dependencies

```python
!pip install -q segmentation-models-pytorch rasterio albumentations opencv-python
```

---

## Training Pipeline

### Configuration

- Optimizer: AdamW  
- Learning Rate: 1e-4  
- Epochs: 15  

---

### Key Steps

- Forward pass  
- Loss computation  
- Gradient backpropagation  
- Weight updates  

---

### Model Saving

```python
torch.save(model.state_dict(), "model.pth")
```

---

## Inference Pipeline

### Test-Time Augmentation (TTA)

- Horizontal flip  
- Vertical flip  

---

### Post-processing

- Thresholding (0.4)  
- Median filtering  
- Noise reduction  

---

### Output Encoding

- Run-Length Encoding (RLE)  

---

## Performance

| Metric | Score |
|--------|------|
| Public Score | 0.189 |

---

## Optimization Strategies

- Data augmentation (spatial + photometric)  
- Advanced models (U-Net++, DeepLabV3+)  
- Training improvements (cosine LR, AMP, gradient clipping)  
- Ensemble learning  
- Threshold tuning  

---

## Production Considerations

- Batch inference pipelines  
- GPU acceleration  
- Deployment via TorchScript / ONNX / FastAPI  
- Monitoring and drift detection  

---

## Limitations

- Limited dataset size  
- Threshold sensitivity  
- Generalization across regions  

---

## Future Work

- Temporal satellite data  
- Transformer models (SegFormer)  
- Real-time deployment  

---

## Team

Code Crew

---

## License

This project is licensed under the ANRF Open License. See LICENSE file for details.
