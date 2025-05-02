import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset paths (update these as needed)
train_dir = '/kaggle/input/alzheimer-5-class-dataset/Alzheimer 5 classes/train'
test_dir = '/kaggle/input/alzheimer-5-class-dataset/Alzheimer 5 classes/test'

# Hyperparameters
batch_size = 64
learning_rate = 0.000005
num_epochs = 50
num_classes = 5
weight_decay = 0.01
patience = 5
image_size = 224

# Preprocessing: Sharpening and CLAHE
def preprocess_image(input_path, output_path, size=(224, 224)):
    os.makedirs(output_path, exist_ok=True)
    for root, _, files in os.walk(input_path):
        for file_name in files:
            if file_name.lower().endswith(('jpg', 'png', 'jpeg')):
                input_file = os.path.join(root, file_name)
                img = cv2.imread(input_file)
                if img is None:
                    print(f"Warning: Unable to read {input_file}. Skipping...")
                    continue
                img_resized = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
                sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                img_sharpened = cv2.filter2D(img_resized, -1, sharpening_kernel)
                img_gray = cv2.cvtColor(img_sharpened, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img_clahe = clahe.apply(img_gray)
                relative_path = os.path.relpath(root, input_path)
                output_subdir = os.path.join(output_path, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_file = os.path.join(output_subdir, file_name)
                cv2.imwrite(output_file, img_clahe)

# Preprocess datasets
output_train_dir = '/kaggle/working/processed_images/train'
output_test_dir = '/kaggle/working/processed_images/test'
preprocess_image(train_dir, output_train_dir, size=(image_size, image_size))
preprocess_image(test_dir, output_test_dir, size=(image_size, image_size))

# Image transformations
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

# Load datasets
full_train_dataset = datasets.ImageFolder(output_train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(output_test_dir, transform=test_transform)

# Split into training and validation
train_size = int(0.9 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Placeholder for model (architecture withheld until paper publication)
class PlaceholderModel(nn.Module):
    def __init__(self, num_classes):
        super(PlaceholderModel, self).__init__()
        # Simplified placeholder: Replace with actual model after publication
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 224 * 224, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Instantiate model
model = PlaceholderModel(num_classes=num_classes).to(device)

# Loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# Tracking variables
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Early stopping variables
best_val_loss = float('inf')
trigger_times = 0

# Training loop
print("Starting training...")
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
    train_loss = running_loss / len(train_loader)
    train_acc = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation phase
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)
    val_loss = running_val_loss / len(val_loader)
    val_acc = correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    scheduler.step(val_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping.")
            break

# Testing phase
print("Testing model...")
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

# Test accuracy
test_accuracy = np.mean(np.array(test_preds) == np.array(test_labels)) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Classification report
print(classification_report(test_labels, test_preds, target_names=full_train_dataset.classes))

# Confusion Matrix
conf_matrix = confusion_matrix(test_labels, test_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=full_train_dataset.classes)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix')
plt.show()

# Plot accuracy and loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, 'bo-', label='Train Acc')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, 'ro-', label='Val Acc')
plt.title('Train vs Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_losses) + 1), train_losses, 'bo-', label='Train Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, 'ro-', label='Val Loss')
plt.title('Train vs Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
