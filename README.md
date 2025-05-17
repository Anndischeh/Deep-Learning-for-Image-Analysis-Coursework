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

## 🚀 How to Run

### In a Linux Environment

### 🔧 Environment Setup (Linux or Linux-like Shell in Windows)

1. **Set up the environment**
   Run the `setup.sh` bash script to create a virtual environment and automatically install all required dependencies:

   ```bash
   bash setup.sh
   ```

   ⚠️ **Note**:

   * The script uses Python 3 with the following command:

     ```bash
     python3 -m venv "Deep_Learning_For_Image_Analyses"
     ```

     If your system uses a different Python version or command (e.g., `python` instead of `python3`), make sure to update this line accordingly.

   * The activation line:

     ```bash
     source "Deep_Learning_For_Image_Analyses/bin/activate"
     ```

     is intended for Unix-based systems (Linux/macOS).
     On Windows, virtual environments use the `Scripts` folder instead of `bin`.

     ✅ If you are using a **Linux-like shell in a Windows environment** (e.g., Git Bash), replace the activation command with:

     ```bash
     source "Deep_Learning_For_Image_Analyses/Scripts/activate"
     ```

2. **Install required packages**
   Once the virtual environment is activated, install all required dependencies listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

---

### 📂 Dataset Path

By default, the dataset path is set inside the code (e.g., `train.py`) as:

```python
'/content/input/annotations/instances_train2017.json'
```

This path is designed for Google Colab or similar environments.
Please **modify it according to your local system setup** so that it correctly points to the location of your `annotations` and image folders.

---

### 🏋️ Train the Model

Execute `train.py` to start training:

```bash
python train.py
```

---
### 🚀 In Google Colab

**1. 📁 Upload Files:**  
Upload all project folders and files as a ZIP archive, **excluding** `App.ipynb`.

---

**2.  📓 Running `App.ipynb` in Google Colab:**

The notebook `App.ipynb` provides a fast, automated way to set up the environment and train the model in Google Colab.

Here’s what each cell does:

1. **Unzip Project Files**

   ```bash
   !unzip /content/Deep_Learning_for_Image_Analysis.zip
2. **Download Dataset**

    ```bash
    !python /content/Deep_Learning_for_Image_Analysis/Utils/downloader.py
This script downloads the COCO dataset and places it in the appropriate directories under input/.

3. **Install Dependencies**

   ```bash
   !pip install -r /content/Deep_Learning_for_Image_Analysis/requirements.txt
Installs all necessary Python packages listed in requirements.txt.

4. **Train the Model**

   ```bash
   !python /content/Deep_Learning_for_Image_Analysis/train.py
Starts the training process using the specified dataset.

***📌 Notes***

- Make sure the ZIP file `Deep_Learning_for_Image_Analysis.zip` is uploaded to your Colab environment before running the notebook.

- You can monitor training progress through the printed logs or by saving `plots/results in the plots/ directory`.

- Modify any paths in the notebook if your folder names or locations are different.
  
------

**▶️ Run Notebook Cells Manually (Optional)**

If you prefer to run steps manually, follow this sequence:

- Create directories for `input` data and `outputs`.

- Download the COCO dataset from the source and save it in the created directory.

- Unzip the dataset.

- Install dependencies and preprocess data.

- Train the model.

- Test on an image.

  -------

**📂 Results**

The trained model's predictions will be saved as images in the plots directory.

**⏳ Performance Note**

Using a GPU A100, the training process takes approximately 7 hours to complete.




---

## 📁 Project Structure
```
├── App.ipynb                                                  #توضیح
├── requirements.txt                                           #توضیح
├── setup.sh                                                   #توضیح
├── train.py                                                   #توضیح  
├── input/                                                     #توضیح
│ └── Input data directory (annotations, images, etc.)
├── Model/                                                     #توضیح
│ ├── loss.py                                                  #توضیح
│ ├── metrics.py                                               #توضیح
│ └── model.py                                                 #توضیح
├── Utils/                                                     #توضیح
│ ├── categories.py                                            #توضیح
│ ├── config.py                                                #توضیح
│ ├── data_loader.py                                           #توضیح
│ ├── downloader.py                                            #توضیح
│ ├── inference.py                                             #توضیح
│ └──  utils.py                                                #توضیح
├── working/                                                   #توضیح
│ ├──lossandmap/                                               #توضیح
│ │ └── landm_YOLOv1_3.csv                                     #توضیح
│ ├── plots/                                                   #توضیح
│ │ ├── batch_sample_images.png
│ │ ├── YOLOv1_3loss_mAP_evolution35.png
│ │ └── predict_images.png
│ ├─ weights/                                                  #توضیح
│ │ └── YOLOv1_3.pt                                            #توضیح

```
