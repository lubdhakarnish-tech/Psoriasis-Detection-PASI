# Psoriasis Detection and PASI Scoring using Deep Learning

This project detects psoriasis using deep learning and estimates severity using PASI scoring.

## 🚀 Features
- Psoriasis classification using ResNet18
- PASI severity scoring (Area, Erythema, Scaling, Thickness)
- Image preprocessing and augmentation
- Lesion segmentation using LAB color space

## 🧠 Model
- Pretrained ResNet18 (Transfer Learning)
- Frozen backbone, trained final layer

## 📊 Dataset

This project uses a dataset of psoriasis and normal skin images.

- Psoriasis images: ~820
- Normal images: ~1651

Due to size and licensing constraints, the dataset is not included in this repository.

You can use your own dataset or publicly available datasets such as:
- ISIC Skin Disease Dataset
- Kaggle Skin Disease datasets


## 📈 Results
- Accuracy: ~90%
- PASI scoring implemented

## 🔮 Future Work
- Use EfficientNet / U-Net
- Better segmentation
- Improved PASI scoring

## 🛠 Tech Stack
- Python
- PyTorch
- OpenCV
- NumPy

## 👨‍💻 Author
Lubdhak Nairith Saha Arnish
