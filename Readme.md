Alzheimer’s Disease Classification Using a Hybrid Deep Learning Model
Abstract
Background: Alzheimer’s disease (AD), a progressive neurodegenerative disorder, demands precise early diagnosis to enable timely interventions. Accurately staging AD is difficult due to the complex combination of local brain changes and widespread connectivity problems.Methods: Traditional CNN and deep learning models struggle to integrate local brain changes with global connectivity patterns, limiting their effectiveness in AD classification. This study proposes a novel deep learning framework with adaptive feature fusion for multi-stage AD classification using T1-weighted MRI scans. The framework integrates a ResNet50-based convolutional neural network (CNN) for extracting fine-grained local features, a Vision Transformer (ViT) for modeling long-range brain connectivity, and an adaptive feature fusion layer for dynamically synthesizing these multi-scale features into a unified representation.Results: Evaluated on the Alzheimer’s 5-Class (AD5C) dataset comprising 2,380 MRI scans, the framework achieves an accuracy of 99.42% (precision: 99.55%, recall: 99.46%, F1-score: 99.50%), surpassing the prior benchmark of 98.24%. Ablation studies underscore the pivotal role of adaptive feature fusion in minimizing misclassifications, while external validation on a four-class dataset confirms robust generalizability.Conclusion: This framework offers transformative potential for clinical AD diagnostics, enhancing early detection and intervention strategies.
Figures

Model Architecture: figures/model_architecture.png (Placeholder schematic of the hybrid model, withheld until publication).
Confusion Matrix: figures/confusion_matrix.png (Test dataset confusion matrix).
Training Plot: figures/training_plot.png (Training and validation accuracy/loss curves).
Results: figures/results.png (Visualization of classification metrics).

Dataset
The dataset used is the Alzheimer 5-class dataset (AD5C), sourced from:

[Insert Paper Citation, e.g., Author(s). (Year). "Title of the Paper." Journal/Conference Name.]


Classes: Mild Demented, Moderate Demented, Non-Demented, Severe Demented, Very Mild Demented.
Size: 2,380 T1-weighted MRI scans.
Structure: Organized into train/ and test/ directories, with subfolders for each class.
Access: Download from the Kaggle link and place in data/, or update train_dir and test_dir in src/main.py to your local dataset path.

Requirements
Install dependencies using Python 3.8+ and the provided requirements.txt:
pip install -r requirements.txt

requirements.txt:
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
opencv-python>=4.5.0
scikit-learn>=0.24.0
matplotlib>=3.3.0

Manual installation:
pip install torch torchvision numpy opencv-python scikit-learn matplotlib

Repository Structure

src/main.py: Main script for preprocessing, training, and evaluation.
figures/:
model_architecture.png: Placeholder model architecture diagram.
confusion_matrix.png: Confusion matrix for test predictions.
training_plot.png: Training/validation accuracy and loss curves.
results.png: Classification metrics visualization.


data/: Placeholder for the dataset (not included; see Dataset section).
requirements.txt: Python dependencies.
LICENSE: MIT License (to be updated post-publication).
README.md: This file.

Model Architecture
The proposed framework integrates a ResNet50-based CNN for local feature extraction, a Vision Transformer (ViT) for long-range connectivity modeling, and a novel adaptive feature fusion layer to combine multi-scale features. Due to the paper being under review, the detailed implementation (HybridModel and AttentionFusion) is withheld to protect intellectual property. A placeholder model is provided in src/main.py for pipeline demonstration.
Figure: Model ArchitectureCaption: Placeholder schematic of the hybrid model. Full details will be released upon publication.
Testing Guidelines
To test the pipeline with the placeholder model:

Prepare the Environment:

Ensure Python 3.8+ and dependencies are installed (see Requirements).
Verify GPU availability for faster training (CUDA-supported GPU recommended).


Download and Set Up the Dataset:

Download the AD5C dataset from Kaggle.
Place the train/ and test/ folders in data/, or modify train_dir and test_dir in src/main.py to point to your dataset location.


Run the Pipeline:
python src/main.py

This executes:

Preprocessing: Applies sharpening, CLAHE, and resizing to MRI scans, saving processed images in /kaggle/working/processed_images/.
Training: Trains the placeholder model with early stopping and learning rate scheduling (50 epochs, batch size 64).
Evaluation: Computes test accuracy, classification report, and generates figures (confusion matrix, training plots).


Expected Outputs:

Model Weights: Saved as best_model.pth (not included in the repository).
Figures: Saved in figures/:
Confusion matrix (confusion_matrix.png).
Training/validation curves (training_plot.png).
Results visualization (results.png, if implemented).


Console Output: Test accuracy, classification report, and epoch-wise training metrics.


Troubleshooting:

Ensure sufficient disk space for processed images (/kaggle/working/).
If CUDA errors occur, set device = torch.device("cpu") in src/main.py.
Verify dataset paths and image formats (JPG, PNG, JPEG).



Results
Evaluated on the AD5C test set (173 samples), the full hybrid model achieves:

Accuracy: 99.42%
Precision: 99.55%
Recall: 99.46%
F1-Score: 99.50%

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

Note: The placeholder model in src/main.py yields lower performance. The above metrics reflect the full hybrid model, as reported in the paper.
Figure: Confusion MatrixCaption: Confusion matrix for test set predictions, showing near-perfect classification.
Figure: Training and Validation CurvesCaption: Training and validation accuracy/loss curves, demonstrating stable convergence.
Figure: Results SummaryCaption: Visualization of precision, recall, and F1-score across classes.
Findings

Superior Performance: The hybrid model outperforms the prior benchmark (98.24%) by leveraging adaptive feature fusion to integrate local and global brain features.
Preprocessing Impact: Sharpening, CLAHE, and data augmentation (random rotation, flipping, color jitter) enhance feature extraction, reducing misclassifications.
Generalizability: External validation on a four-class dataset confirms the model’s robustness across different AD classification tasks.
Ablation Insights: Adaptive feature fusion is critical for minimizing errors, particularly for challenging classes like Mild Demented.
Limitations: The placeholder model lacks the hybrid architecture’s sophistication, resulting in lower performance. Class imbalance (e.g., fewer VeryMildDemented samples) may still pose challenges.

Note
The model architecture (HybridModel, AttentionFusion) and trained weights are withheld until the paper is published to protect novel contributions. The provided src/main.py includes a placeholder model to demonstrate the pipeline. Post-publication, the repository will be updated with:

Full model implementation.
Detailed architecture diagram (figures/model_architecture.png).
Trained weights (if permitted by the dataset license).

For reviewer access to the full code or weights, contact [Your Email]. Updates will be announced post-publication.
License
This project is licensed under the MIT License (see LICENSE). The full code release post-publication will also be under this license.
Contact
For questions or collaboration inquiries, contact [Your Name] at [Your Email].
Acknowledgments

The AD5C dataset providers and the referenced paper authors.
The open-source community for tools like PyTorch, OpenCV, and scikit-learn.
Kaggle for hosting the dataset.

