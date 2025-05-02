Alzheimer’s Disease Classification 🧠

This repository contains the code for a deep learning pipeline to classify Alzheimer’s disease stages using MRI scans. The model achieves 99.42% accuracy on the Alzheimer’s 5-Class (AD5C) dataset. The architecture is withheld until paper publication, with a placeholder provided for reproducibility.
📝 Abstract
We propose a hybrid deep learning model for multi-stage Alzheimer’s disease classification using T1-weighted MRI scans from the AD5C dataset. Combining ResNet50 and Vision Transformer (ViT) with adaptive feature fusion, it achieves 99.42% accuracy, surpassing the 98.24% benchmark. This repository includes preprocessing, training, and evaluation code, with the full model to be released post-publication.
📊 Figures

figures/model_architecture.png: Placeholder model schematic (full details post-publication).
figures/confusion_matrix.png: Test set confusion matrix.
figures/training_plot.png: Training/validation accuracy and loss curves.
figures/results.png: Classification metrics visualization.

🗄️ Dataset
Source: Alzheimer 5-class dataset (AD5C)Citation:  

[Insert Citation, e.g., Author(s). (Year). "Title of the Paper." Journal/Conference Name.]


Classes: Mild Demented, Moderate Demented, Non-Demented, Severe Demented, Very Mild Demented
Size: 2,380 MRI scans
Setup: Download from Kaggle, place in data/, or update paths in src/main.py.

🛠️ Requirements
Install dependencies using Python 3.8+:
pip install -r requirements.txt

requirements.txt:
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
opencv-python>=4.5.0
scikit-learn>=0.24.0
matplotlib>=3.3.0

Or manually:
pip install torch torchvision numpy opencv-python scikit-learn matplotlib

📂 Repository Structure
alzheimer-classification/
├── src/
│   └── main.py               # Preprocessing, training, evaluation
├── figures/
│   ├── model_architecture.png
│   ├── confusion_matrix.png
│   ├── training_plot.png
│   └── results.png
├── data/                     # Dataset placeholder
├── requirements.txt          # Dependencies
├── LICENSE                   # MIT License
└── README.md                 # This file

🧠 Model Architecture
The model integrates ResNet50, ViT, and an adaptive feature fusion layer for superior AD classification. To protect intellectual property, the implementation (HybridModel, AttentionFusion) is withheld until publication. A placeholder model is provided in src/main.py.
Placeholder schematic. Full architecture will be shared post-publication.
🚀 Testing Guidelines

Setup Environment:

Install dependencies (see Requirements).
Use a CUDA-supported GPU for faster training, or set device = "cpu" in src/main.py.


Prepare Dataset:

Download AD5C dataset from Kaggle.
Place train/ and test/ in data/, or update paths in src/main.py.


Run Pipeline:
python src/main.py


Preprocessing: Sharpening, CLAHE, resizing.
Training: 50 epochs, batch size 64, early stopping.
Evaluation: Generates accuracy, classification report, and figures.


Outputs:

Weights: best_model.pth (not shared).
Figures: Saved in figures/ (confusion matrix, training plots, results).
Console: Test metrics and training logs.


Troubleshooting:

Ensure disk space for /kaggle/working/processed_images/.
Verify image formats (JPG, PNG, JPEG).
Check dataset paths in src/main.py.



📈 Results
The full model achieves 99.42% accuracy on the AD5C test set (173 samples):
Classification Report:
                  precision    recall  f1-score   support
    MildDemented       1.00      0.98      0.99        49
ModerateDemented       1.00      1.00      1.00        42
     NonDemented       0.96      1.00      0.98        22
  SevereDemented       1.00      1.00      1.00        47
VeryMildDemented       1.00      1.00      1.00        13
        accuracy                           0.99       173
       macro avg       0.99      1.00      0.99       173
    weighted avg       0.99      0.99      0.99       173


Precision: 99.55%
Recall: 99.46%
F1-Score: 99.50%

Note: The placeholder model yields lower performance. Above metrics reflect the full hybrid model.
Test set confusion matrix.
Training/validation accuracy and loss curves.
Precision, recall, and F1-score visualization.
🔍 Findings

Performance: Outperforms 98.24% benchmark via adaptive feature fusion.
Preprocessing: Sharpening, CLAHE, and augmentations reduce misclassifications.
Generalizability: Validated on a four-class dataset, showing robustness.
Ablation: Feature fusion is key for distinguishing Mild Demented cases.
Limitations: Placeholder model is less effective; class imbalance may impact results.

⚠️ Note
The model architecture and weights are withheld until paper publication. src/main.py includes a placeholder model. Post-publication, we’ll release:

Full HybridModel and AttentionFusion code.
Updated figures/model_architecture.png.
Weights (if dataset license allows).

Contact [Your Email] for reviewer access to full code/weights.
📜 License
MIT License (full code will be MIT post-publication).
📬 Contact
[Your Name] - [Your Email]
🙏 Acknowledgments

AD5C dataset providers and paper authors.
PyTorch, OpenCV, and scikit-learn communities.
Kaggle for dataset hosting.

