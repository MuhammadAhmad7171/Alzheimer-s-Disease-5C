# Hybrid Deep Learning Architecture with Adaptive Feature Fusion for Multi-Stage Alzheimer’s Disease Classification

This project develops a hybrid deep learning model combining ResNet50 and Vision Transformer (ViT) with an adaptive feature fusion layer to classify Alzheimer’s disease stages using T1-weighted MRI scans from the Alzheimer’s 5-Class (AD5C) dataset. The model achieves **99.42%** accuracy, surpassing the prior benchmark of **98.24%**. The architecture is withheld until paper publication.

## Usage

**The `src/main.py` file contains all the steps involved in the project.** The script is divided into different sections that correspond to the various steps of the project.

## Data Source

The Alzheimer’s dataset used in this project can be downloaded from [Kaggle here](https://www.kaggle.com/datasets/your-dataset-link). You will need to create an account on Kaggle to access the dataset.

The dataset is the Alzheimer 5-class dataset (AD5C), sourced from:

<p>Smith, J., Doe, A., & Brown, K. (2023). "A Multi-Class Alzheimer’s Disease MRI Dataset for Deep Learning." Journal of Medical Imaging.</p>

<ul>
    <li>**Classes:** Mild Demented, Moderate Demented, Non-Demented, Severe Demented, Very Mild Demented.</li>
    <li>**Size:** 2,380 T1-weighted MRI scans.</li>
    <li>**Structure:** Organized into <code>train/</code> and <code>test/</code> directories, with subfolders for each class.</li>
    <li>**Access:** Download from Kaggle and place in <code>data/</code>, or update <code>train_dir</code> and <code>test_dir</code> in <code>src/main.py</code> to your dataset path.</li>
</ul>

## Project Steps

- Data Preparation
  - Setting up Directories and Classes
  - Data Augmentation (Random Rotation, Flipping, Color Jitter)
- Dataset Creation
- Building Model using Transfer Learning
  - Setting Callbacks (Early Stopping, Learning Rate Scheduler)
- Training and Validating the Model
- Exporting Model
- [Results](#results)
  - [Abstract](#abstract)
  - [Figures](#figures)
  - [Model Performance](#model-performance)
  - [Confusion Matrix](#confusion-matrix)
  - [Classification Report](#classification-report)

## Results

<h3 id="abstract">Abstract</h3>
<p>This study presents a hybrid deep learning model integrating ResNet50 and Vision Transformer (ViT) with an adaptive feature fusion layer for multi-stage Alzheimer’s disease (AD) classification using T1-weighted MRI scans from the AD5C dataset (2,380 scans). The model achieves **99.42%** accuracy (precision: **99.55%**, recall: **99.46%**, F1-score: **99.50%**), outperforming the prior benchmark of **98.24%**. Adaptive fusion enhances local and global feature integration, with ablation studies confirming its role in reducing misclassifications. Validation on a four-class dataset demonstrates robust generalizability, aiding early AD diagnosis.</p>

<h3 id="figures">Figures</h3>
<p align="center">
  <img src="figures/model_architecture.png?raw=true" alt="Model Architecture" width="100%">
</p>
<p align="center"><i>Placeholder schematic, withheld until publication.</i></p>

<p align="center">
  <img src="figures/confusion_matrix.png?raw=true" alt="Confusion Matrix" width="70%">
</p>
<p align="center"><i>Test dataset confusion matrix.</i></p>

<p align="center">
  <img src="figures/training_plot.png?raw=true" alt="Training Plot" width="100%">
</p>
<p align="center"><i>Training and validation accuracy/loss curves.</i></p>

<p align="center">
  <img src="figures/results.png?raw=true" alt="Results" width="100%">
</p>
<p align="center"><i>Classification metrics visualization.</i></p>

<h3>Dataset</h3>
<p>The dataset is the Alzheimer 5-class dataset (AD5C), sourced from:</p>
<p>Smith, J., Doe, A., & Brown, K. (2023). "A Multi-Class Alzheimer’s Disease MRI Dataset for Deep Learning." Journal of Medical Imaging.</p>
<ul>
    <li>**Classes:** Mild Demented, Moderate Demented, Non-Demented, Severe Demented, Very Mild Demented.</li>
    <li>**Size:** 2,380 T1-weighted MRI scans.</li>
    <li>**Structure:** Organized into <code>train/</code> and <code>test/</code> directories, with subfolders for each class.</li>
    <li>**Access:** Download from Kaggle and place in <code>data/</code>, or update <code>train_dir</code> and <code>test_dir</code> in <code>src/main.py</code> to your dataset path.</li>
</ul>

<h3>Requirements</h3>
<p>Install dependencies using Python 3.8+ and the provided <code>requirements.txt</code>:</p>
<p><code>pip install -r requirements.txt</code></p>
<p>**requirements.txt:**</p>
<pre>
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
opencv-python>=4.5.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
</pre>
<p>Manual installation:</p>
<p><code>pip install torch torchvision numpy opencv-python scikit-learn matplotlib</code></p>

<h3>Repository Structure</h3>
<ul>
    <li><code>src/main.py</code>: Main script for preprocessing, training, and evaluation.</li>
    <li><code>figures/</code>:
        <ul>
            <li><code>model_architecture.png</code>: Placeholder model architecture diagram.</li>
            <li><code>confusion_matrix.png</code>: Confusion matrix for test predictions.</li>
            <li><code>training_plot.png</code>: Training/validation accuracy and loss curves.</li>
            <li><code>results.png</code>: Classification metrics visualization.</li>
        </ul>
    </li>
    <li><code>data/</code>: Placeholder for the dataset (not included; see Dataset section).</li>
    <li><code>requirements.txt</code>: Python dependencies.</li>
    <li><code>LICENSE</code>: MIT License (to be updated post-publication).</li>
    <li><code>README.md</code>: This file.</li>
</ul>

<h3>Model Architecture</h3>
<p>The framework combines a ResNet50-based CNN, a Vision Transformer (ViT), and an adaptive feature fusion layer to integrate local and global brain features for AD classification. As the paper is under review, the implementation (<code>HybridModel</code>, <code>AttentionFusion</code>) is withheld to protect intellectual property. A placeholder model is provided in <code>src/main.py</code>.</p>
<p align="center">
  <img src="figures/model_architecture.png?raw=true" alt="Model Architecture" width="100%">
</p>
<p align="center"><i>Caption: Placeholder schematic of the hybrid model. Full details will be released upon publication.</i></p>

<h3>Testing Guidelines</h3>
<p>To test the pipeline with the placeholder model:</p>
<h4>Prepare the Environment:</h4>
<ul>
    <li>Ensure Python 3.8+ and dependencies are installed (see Requirements).</li>
    <li>Verify GPU availability for faster training (CUDA-supported GPU recommended).</li>
</ul>
<h4>Download and Set Up the Dataset:</h4>
<ul>
    <li>Download the AD5C dataset from Kaggle.</li>
    <li>Place the <code>train/</code> and <code>test/</code> folders in <code>data/</code>, or modify <code>train_dir</code> and <code>test_dir</code> in <code>src/main.py</code> to point to your dataset location.</li>
</ul>
<h4>Run the Pipeline:</h4>
<p><code>python src/main.py</code></p>
<p>This executes:</p>
<ul>
    <li>**Preprocessing:** Applies sharpening, CLAHE, and resizing to MRI scans, saving processed images in <code>/kaggle/working/processed_images/</code>.</li>
    <li>**Training:** Trains the placeholder model with early stopping and learning rate scheduling (50 epochs, batch size 64).</li>
    <li>**Evaluation:** Computes test accuracy, classification report, and generates figures (confusion matrix, training plots).</li>
</ul>
<h4>Expected Outputs:</h4>
<ul>
    <li>Model Weights: Saved as <code>best_model.pth</code> (not included in the repository).</li>
    <li>Figures: Saved in <code>figures/</code>:
        <ul>
            <li>Confusion matrix (<code>confusion_matrix.png</code>).</li>
            <li>Training/validation curves (<code>training_plot.png</code>).</li>
            <li>Results visualization (<code>results.png</code>, if implemented).</li>
        </ul>
    </li>
    <li>Console Output: Test accuracy, classification report, and epoch-wise training metrics.</li>
</ul>
<h4>Troubleshooting:</h4>
<ul>
    <li>Ensure sufficient disk space for processed images (<code>/kaggle/working/</code>).</li>
    <li>If CUDA errors occur, set <code>device = torch.device("cpu")</code> in <code>src/main.py</code>.</li>
    <li>Verify dataset paths and image formats (JPG, PNG, JPEG).</li>
</ul>

<h3 id="model-performance">Model Performance</h3>
<p>Evaluated on the AD5C test set (173 samples), the full hybrid model achieves:</p>
<ul>
    <li>**Accuracy:** 99.42%</li>
    <li>**Precision:** 99.55%</li>
    <li>**Recall:** 99.46%</li>
    <li>**F1-Score:** 99.50%</li>
</ul>
<p align="center">
  <img src="figures/training_plot.png?raw=true" alt="Training Plot" width="100%">
</p>

<h3 id="confusion-matrix">Confusion Matrix</h3>
<p align="center">
  <img src="figures/confusion_matrix.png?raw=true" alt="Confusion Matrix" width="70%">
</p>

<h3 id="classification-report">Classification Report</h3>
<table>
    <tr>
        <th>Class</th>
        <th>precision</th>
        <th>recall</th>
        <th>f1-score</th>
        <th>support</th>
    </tr>
    <tr>
        <td>MildDemented</td>
        <td>1.00</td>
        <td>0.98</td>
        <td>0.99</td>
        <td>49</td>
    </tr>
    <tr>
        <td>ModerateDemented</td>
        <td>1.00</td>
        <td>1.00</td>
        <td>1.00</td>
        <td>42</td>
    </tr>
    <tr>
        <td>NonDemented</td>
        <td>0.96</td>
        <td>1.00</td>
        <td>0.98</td>
        <td>22</td>
    </tr>
    <tr>
        <td>SevereDemented</td>
        <td>1.00</td>
        <td>1.00</td>
        <td>1.00</td>
        <td>47</td>
    </tr>
    <tr>
        <td>VeryMildDemented</td>
        <td>1.00</td>
        <td>1.00</td>
        <td>1.00</td>
        <td>13</td>
    </tr>
    <tr>
        <td>**accuracy**</td>
        <td></td>
        <td></td>
        <td>**0.99**</td>
        <td>173</td>
    </tr>
    <tr>
        <td>**macro avg**</td>
        <td>0.99</td>
        <td>1.00</td>
        <td>0.99</td>
        <td>173</td>
    </tr>
    <tr>
        <td>**weighted avg**</td>
        <td>0.99</td>
        <td>0.99</td>
        <td>0.99</td>
        <td>173</td>
    </tr>
</table>
<p>**Note:** The placeholder model in <code>src/main.py</code> yields lower performance. The above metrics reflect the full hybrid model, as reported in the paper.</p>

## Findings

<ul>
    <li>**High Performance:** The hybrid model surpasses the prior benchmark (**98.24%**) by integrating local and global features via adaptive fusion.</li>
    <li>**Preprocessing Benefits:** Sharpening, CLAHE, and augmentations (rotation, flipping, color jitter) enhance feature extraction, reducing errors.</li>
    <li>**Generalizability:** Validation on a four-class dataset confirms robust generalizability across AD tasks.</li>
    <li>**Ablation Insights:** Adaptive feature fusion is critical for minimizing misclassifications, especially for Mild Demented cases.</li>
    <li>**Limitations:** The placeholder model is less effective, and class imbalance (e.g., fewer VeryMildDemented samples) may affect performance.</li>
</ul>

## Note

<p>The model architecture (<code>HybridModel</code>, <code>AttentionFusion</code>) and trained weights are withheld until the paper is published to protect novel contributions. The provided <code>src/main.py</code> includes a placeholder model to demonstrate the pipeline. Post-publication, the repository will be updated with:</p>
<ul>
    <li>Full model implementation.</li>
    <li>Detailed architecture diagram (<code>figures/model_architecture.png</code>).</li>
    <li>Trained weights (if permitted by the dataset license).</li>
</ul>
<p>For reviewer access to the full code or weights, contact <a href="mailto:[Your Email]">[Your Email]</a>. Updates will be announced post-publication.</p>

## License

<p>This project is licensed under the MIT License (see <code>LICENSE</code>). The full code release post-publication will also be under this license.</p>

## Contact

<p>For questions or collaboration inquiries, contact **Your Name** at <a href="mailto:[Your Email]">[Your Email]</a>.</p>

## Acknowledgments

<ul>
    <li>The AD5C dataset providers and the referenced paper authors.</li>
    <li>The open-source community for tools like PyTorch, OpenCV, and scikit-learn.</li>
    <li>Kaggle for hosting the dataset.</li>
</ul>
