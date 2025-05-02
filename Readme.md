Alzheimer’s Disease Classification Using a Hybrid Deep Learning Model
Abstract
This repository contains the code for a deep learning pipeline designed to classify Alzheimer’s disease stages using brain MRI images from a 5-class dataset. The approach employs a novel hybrid model combining convolutional neural networks and vision transformers with an attention-based fusion mechanism. The pipeline includes advanced image preprocessing, model training, and evaluation. To protect intellectual property prior to paper publication, the model architecture details are withheld, with a placeholder model provided for reproducibility. Full code will be released upon publication.
Dataset
The dataset used in this project is the Alzheimer 5-class dataset, sourced from the paper:

[Insert Paper Citation, e.g., Author(s). (Year). "Title of the Paper." Journal/Conference Name.]


Classes: 5 (e.g., Non-Demented, Very Mild Demented, Mild Demented, Moderate Demented, Severe Demented).
Structure: The dataset is split into train/ and test/ directories, with subfolders for each class.
Access: Download the dataset from the Kaggle link above and place it in the data/ folder, or update the paths in src/main.py to point to your local dataset location.

Requirements
To run the code, install the required Python packages listed below. Python 3.8 or higher is recommended.
pip install -r requirements.txt

requirements.txt:
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
opencv-python>=4.5.0
scikit-learn>=0.24.0
matplotlib>=3.3.0

Alternatively, install manually:
pip install torch torchvision numpy opencv-python scikit-learn matplotlib

Repository Structure

src/main.py: Main script for preprocessing, training, and evaluating the model.
figures/:
model_architecture.png: Diagram of the hybrid model (placeholder until publication).
confusion_matrix.png: Confusion matrix for test dataset predictions.
training_plot.png: Training and validation accuracy/loss curves.
results.png: Visualization of classification performance metrics.


data/: Placeholder for the dataset (not included; see Dataset section).
requirements.txt: List of Python dependencies.
LICENSE: MIT License (to be updated with full code release).
README.md: This file.

Model Architecture
The proposed model is a hybrid architecture combining a convolutional neural network (ResNet-50) and a vision transformer (ViT) with a novel attention-based fusion layer to integrate features from both backbones. Due to the paper being under review, the detailed implementation of the model architecture (HybridModel and AttentionFusion) is withheld to protect intellectual property. A placeholder model is provided in src/main.py to demonstrate the pipeline's functionality.
Figure: Model ArchitectureNote: The above figure is a placeholder schematic. The full architecture diagram and implementation will be released upon paper publication.
Running the Code

Prepare the Dataset:

Download the Alzheimer 5-class dataset from Kaggle.
Place it in data/ or update the train_dir and test_dir paths in src/main.py.


Install Dependencies:
pip install -r requirements.txt


Run the Pipeline:
python src/main.py

This script performs:

Image preprocessing (sharpening, CLAHE, resizing).
Data loading with augmentation for training.
Training with early stopping and learning rate scheduling.
Evaluation on the test set, including classification metrics and confusion matrix generation.


Output:

Figures (e.g., confusion matrix, training plots) are saved in figures/.
Model weights are saved as best_model.pth (not included in the repository).



Results
The pipeline was evaluated on the test split of the Alzheimer 5-class dataset. Key performance metrics include accuracy, precision, recall, and F1-score for each class. The placeholder model achieves lower performance than the original hybrid model, but the pipeline demonstrates the preprocessing and evaluation process.
Figure: Confusion MatrixCaption: Confusion matrix showing the model's performance on the test dataset.
Figure: Training and Validation CurvesCaption: Training and validation accuracy/loss curves over epochs.
Figure: Results SummaryCaption: Visualization of classification metrics (accuracy, precision, recall, F1-score).
Sample Metrics (Placeholder Model)
Note: These metrics are based on the placeholder model and do not reflect the hybrid model's performance, which will be detailed in the paper.



Class
Precision
Recall
F1-Score
Support



Non-Demented
0.XX
0.XX
0.XX
XXX


Very Mild Demented
0.XX
0.XX
0.XX
XXX


Mild Demented
0.XX
0.XX
0.XX
XXX


Moderate Demented
0.XX
0.XX
0.XX
XXX


Severe Demented
0.XX
0.XX
0.XX
XXX


Accuracy: XX.XX% (placeholder model)
Findings

Preprocessing Impact: The combination of sharpening, CLAHE, and data augmentation (random rotation, flipping, color jitter) significantly improved model robustness, particularly for distinguishing between closely related classes (e.g., Very Mild vs. Mild Demented).
Placeholder Model Limitations: The simplified placeholder model struggles with complex feature extraction, highlighting the importance of the hybrid architecture (to be released post-publication).
Class Imbalance: Some classes (e.g., Severe Demented) have fewer samples, which affects performance. Techniques like weighted loss or oversampling could further improve results.
Future Work: The full hybrid model, with its attention-based fusion, achieves superior performance, as will be detailed in the paper. Post-publication, we plan to explore additional datasets and fine-tuning strategies.

Note
The model architecture and trained weights are withheld until the paper is published to protect the novel contributions. The provided code includes a placeholder model to demonstrate the pipeline's functionality. Upon publication, the repository will be updated with:

The complete HybridModel and AttentionFusion implementations.
The full model architecture diagram.
Trained model weights (if permitted by the dataset license).

For reviewer access to the full code or weights, please contact [Your Email]. Stay tuned for updates after publication!
License
This project is licensed under the MIT License (see LICENSE). The full code release post-publication will also be under this license.
Contact
For questions or collaboration inquiries, please contact [Your Name] at [Your Email].
Acknowledgments

The Alzheimer 5-class dataset providers and the referenced paper authors.
The open-source community for tools like PyTorch, OpenCV, and scikit-learn.

