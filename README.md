# YOLOv1 Object Detection Enhancement

This repository presents the implementation and enhancement of the YOLOv1 (You Only Look Once) object detection model, aiming to develop a computationally lighter version while maintaining effective detection capabilities.  
The model is trained using the [COCO 2017 dataset](http://images.cocodataset.org/annotations/annotations_trainval2017.zip), and training progress is tracked using [Weights & Biases (WandB)](https://wandb.ai/anndischeh-univ-/Deep%20Learning%20for%20image%20Analysis).

Model Weights: Due to GitHub's file size limitations (>100MB), the trained model (YOLOv1_3.pt, 430MB) is stored externally: [Download YOLOv1_3.pt](https://drive.google.com/file/d/1utKK72AD_tIYivgUkhTo8er-6AIp_3JC/view?usp=drive_link)


---

## 📂 Essential Links and Data

- **Dataset**:  
  Download the COCO 2017 dataset annotations:  
  🔗 [Download COCO 2017 Annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

- **Training Results**:  
  Training is managed through `train.py`, and results (loss curves, evaluation metrics, and prediction samples) are logged using WandB:  
  🔗 [View WandB Project](https://wandb.ai/anndischeh-univ-/Deep%20Learning%20for%20image%20Analysis)

- **Model Weights**:  
  Due to GitHub's file size limitations (>100MB), the final trained model (`YOLOv1_3.pt`, 430MB) is stored externally:  
  🔗 [Download YOLOv1_3.pt](https://drive.google.com/file/d/1utKK72AD_tIYivgUkhTo8er-6AIp_3JC/view?usp=drive_link)

---

## 📁 Project Structure
 C:. │ App.ipynb # Jupyter notebook for experiments and visualization │ requirements.txt # Required Python
packages │ setup.sh # Bash script to set up virtual environment and install requirements │ train.py # Main training loop │ ├── input/ # Input data directory (annotations, images, etc.) │ ├── Model/ # Model implementation and related components │ │ loss.py # YOLOv1 custom loss function │ │ metrics.py # Evaluation metrics (e.g., mAP calculation) │ │ model.py # YOLOv1 architecture │ ├── Utils/ # Utility functions and helper scripts │ │ categories.py # Category definitions │ │ config.py # Configuration settings │ │ data_loader.py # Data loading and augmentation │ │ downloader.py # Dataset download utilities │ │ inference.py # Inference scripts │ │ utils.py # General utility functions │ └── pycache/ # Cached Python files │ └── working/ # Training outputs ├── lossandmap/ # Training loss and mAP evolution CSV │ └── landm_YOLOv1_3.csv ├── plots/ # Visualization of predictions and training progress │ ├── batch_sample_images.png │ ├── predict_images.png │ └── YOLOv1_3loss_mAP_evolution35.png └── weights/ # Checkpoints (links instead of large files) └── LINKofYOLOv1_3.txt
