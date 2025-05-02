<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Alzheimer’s Disease Classification Using a Hybrid Deep Learning Model</title>
</head>
<body>
    <h1>Alzheimer’s Disease Classification Using a Hybrid Deep Learning Model</h1>

    <h2>Abstract</h2>
    <p>This study introduces a hybrid deep learning framework for multi-stage Alzheimer’s disease (AD) classification using T1-weighted MRI scans from the Alzheimer’s 5-Class (AD5C) dataset. Combining ResNet50 and Vision Transformer (ViT) with adaptive feature fusion, the model achieves <strong>99.42%</strong> accuracy, surpassing the prior benchmark of <strong>98.24%</strong>. The code demonstrates preprocessing, training, and evaluation, with the model architecture withheld until paper publication.</p>

    <h2>Figures</h2>
    <ul>
        <li>Model Architecture: <img src="figures/model_architecture.png?raw=true" alt="Model Architecture" width="100%"><br><i>Placeholder schematic, withheld until publication.</i></li>
        <li>Confusion Matrix: <img src="figures/confusion_matrix.png?raw=true" alt="Confusion Matrix" width="70%"><br><i>Test dataset confusion matrix.</i></li>
        <li>Training Plot: <img src="figures/training_plot.png?raw=true" alt="Training Plot" width="100%"><br><i>Training and validation accuracy/loss curves.</i></li>
        <li>Results: <img src="figures/results.png?raw=true" alt="Results" width="100%"><br><i>Classification metrics visualization.</i></li>
    </ul>

    <h2>Dataset</h2>
    <p>The dataset is the Alzheimer 5-class dataset (AD5C), sourced from:</p>
    <p>[Insert Paper Citation, e.g., Author(s). (Year). "Title of the Paper." Journal/Conference Name.]</p>
    <ul>
        <li><strong>Classes:</strong> Mild Demented, Moderate Demented, Non-Demented, Severe Demented, Very Mild Demented.</li>
        <li><strong>Size:</strong> 2,380 T1-weighted MRI scans.</li>
        <li><strong>Structure:</strong> Organized into <code>train/</code> and <code>test/</code> directories, with subfolders for each class.</li>
        <li><strong>Access:</strong> Download from Kaggle and place in <code>data/</code>, or update <code>train_dir</code> and <code>test_dir</code> in <code>src/main.py</code> to your dataset path.</li>
    </ul>

    <h2>Requirements</h2>
    <p>Install dependencies using Python 3.8+ and the provided <code>requirements.txt</code>:</p>
    <p><code>pip install -r requirements.txt</code></p>
    <p><strong>requirements.txt:</strong></p>
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

    <h2>Repository Structure</h2>
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

    <h2>Model Architecture</h2>
    <p>The framework combines a ResNet50-based CNN, a Vision Transformer (ViT), and an adaptive feature fusion layer to integrate local and global brain features for AD classification. As the paper is under review, the implementation (<code>HybridModel</code>, <code>AttentionFusion</code>) is withheld to protect intellectual property. A placeholder model is provided in <code>src/main.py</code>.</p>
    <p>Figure: <img src="figures/model_architecture.png?raw=true" alt="Model Architecture" width="100%"><br><i>Caption: Placeholder schematic of the hybrid model. Full details will be released upon publication.</i></p>

    <h2>Testing Guidelines</h2>
    <p>To test the pipeline with the placeholder model:</p>
    <h3>Prepare the Environment:</h3>
    <ul>
        <li>Ensure Python 3.8+ and dependencies are installed (see Requirements).</li>
        <li>Verify GPU availability for faster training (CUDA-supported GPU recommended).</li>
    </ul>
    <h3>Download and Set Up the Dataset:</h3>
    <ul>
        <li>Download the AD5C dataset from Kaggle.</li>
        <li>Place the <code>train/</code> and <code>test/</code> folders in <code>data/</code>, or modify <code>train_dir</code> and <code>test_dir</code> in <code>src/main.py</code> to point to your dataset location.</li>
    </ul>
    <h3>Run the Pipeline:</h3>
    <p><code>python src/main.py</code></p>
    <p>This executes:</p>
    <ul>
        <li><strong>Preprocessing:</strong> Applies sharpening, CLAHE, and resizing to MRI scans, saving processed images in <code>/kaggle/working/processed_images/</code>.</li>
        <li><strong>Training:</strong> Trains the placeholder model with early stopping and learning rate scheduling (50 epochs, batch size 64).</li>
        <li><strong>Evaluation:</strong> Computes test accuracy, classification report, and generates figures (confusion matrix, training plots).</li>
    </ul>
    <h3>Expected Outputs:</h3>
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
    <h3>Troubleshooting:</h3>
    <ul>
        <li>Ensure sufficient disk space for processed images (<code>/kaggle/working/</code>).</li>
        <li>If CUDA errors occur, set <code>device = torch.device("cpu")</code> in <code>src/main.py</code>.</li>
        <li>Verify dataset paths and image formats (JPG, PNG, JPEG).</li>
    </ul>

    <h2>Results</h2>
    <p>Evaluated on the AD5C test set (173 samples), the full hybrid model achieves:</p>
    <ul>
        <li><strong>Accuracy:</strong> 99.42%</li>
        <li><strong>Precision:</strong> 99.55%</li>
        <li><strong>Recall:</strong> 99.46%</li>
        <li><strong>F1-Score:</strong> 99.50%</li>
    </ul>
    <h3>Classification Report:</h3>
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
            <td><strong>accuracy</strong></td>
            <td></td>
            <td></td>
            <td><strong>0.99</strong></td>
            <td>173</td>
        </tr>
        <tr>
            <td><strong>macro avg</strong></td>
            <td>0.99</td>
            <td>1.00</td>
            <td>0.99</td>
            <td>173</td>
        </tr>
        <tr>
            <td><strong>weighted avg</strong></td>
            <td>0.99</td>
            <td>0.99</td>
            <td>0.99</td>
            <td>173</td>
        </tr>
    </table>
    <p><strong>Note:</strong> The placeholder model in <code>src/main.py</code> yields lower performance. The above metrics reflect the full hybrid model, as reported in the paper.</p>
    <h3>Figure: Confusion Matrix</h3>
    <p><img src="figures/confusion_matrix.png?raw=true" alt="Confusion Matrix" width="70%"><br><i>Caption: Confusion matrix for test set predictions, showing near-perfect classification.</i></p>
    <h3>Figure: Training and Validation Curves</h3>
    <p><img src="figures/training_plot.png?raw=true" alt="Training and Validation Curves" width="100%"><br><i>Caption: Training and validation accuracy/loss curves, demonstrating stable convergence.</i></p>
    <h3>Figure: Results Summary</h3>
    <p><img src="figures/results.png?raw=true" alt="Results Summary" width="100%"><br><i>Caption: Visualization of precision, recall, and F1-score across classes.</i></p>

    <h2>Findings</h2>
    <ul>
        <li><strong>High Performance:</strong> The hybrid model surpasses the prior benchmark (<strong>98.24%</strong>) by integrating local and global features via adaptive fusion.</li>
        <li><strong>Preprocessing Benefits:</strong> Sharpening, CLAHE, and augmentations (rotation, flipping, color jitter) enhance feature extraction, reducing errors.</li>
        <li><strong>Generalizability:</strong> Validation on a four-class dataset confirms robustness across AD tasks.</li>
        <li><strong>Ablation Insights:</strong> Adaptive feature fusion is critical for minimizing misclassifications, especially for Mild Demented cases.</li>
        <li><strong>Limitations:</strong> The placeholder model is less effective, and class imbalance (e.g., fewer VeryMildDemented samples) may affect performance.</li>
    </ul>

    <h2>Note</h2>
    <p>The model architecture (<code>HybridModel</code>, <code>AttentionFusion</code>) and trained weights are withheld until the paper is published to protect novel contributions. The provided <code>src/main.py</code> includes a placeholder model to demonstrate the pipeline. Post-publication, the repository will be updated with:</p>
    <ul>
        <li>Full model implementation.</li>
        <li>Detailed architecture diagram (<code>figures/model_architecture.png</code>).</li>
        <li>Trained weights (if permitted by the dataset license).</li>
    </ul>
    <p>For reviewer access to the full code or weights, contact <a href="mailto:[Your Email]">[Your Email]</a>. Updates will be announced post-publication.</p>

    <h2>License</h2>
    <p>This project is licensed under the MIT License (see <code>LICENSE</code>). The full code release post-publication will also be under this license.</p>

    <h2>Contact</h2>
    <p>For questions or collaboration inquiries, contact <strong>[Your Name]</strong> at <a href="mailto:[Your Email]">[Your Email]</a>.</p>

    <h2>Acknowledgments</h2>
    <ul>
        <li>The AD5C dataset providers and the referenced paper authors.</li>
        <li>The open-source community for tools like PyTorch, OpenCV, and scikit-learn.</li>
        <li>Kaggle for hosting the dataset.</li>
    </ul>
</body>
</html>
