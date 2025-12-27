import os
from PIL import Image
import torchvision.transforms as transforms
import random

# Define your root dataset path
input_root = '/content/colored_images'
output_root = '/content/augmented_colored'
os.makedirs(output_root, exist_ok=True)

# Augmentation transform
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomResizedCrop(size=299, scale=(0.9, 1.1)),
])

# Target class counts
target_count = 1000  # You can adjust this per class

for class_label in ['1', '3', '4']:
    input_dir = os.path.join(input_root, class_label)
    output_dir = os.path.join(output_root, class_label)
    os.makedirs(output_dir, exist_ok=True)

    images = os.listdir(input_dir)
    count = len(images)
    idx = 0

    while count < target_count:
        img_name = images[idx % len(images)]
        img_path = os.path.join(input_dir, img_name)

        with Image.open(img_path).convert('RGB') as img:
            aug_img = augmentation(img)
            save_path = os.path.join(output_dir, f"aug_{count}.png")
            aug_img.save(save_path)
            count += 1
        idx += 1

    print(f"Class {class_label} augmented to {target_count} images.")
	
	
	
	
	
	
	
	
	import os
import random

# Define the input directory for class '0'
class_0_dir = '/content/colored_images/0'
target_count = 1000  # Desired number of images

# List all image files in the class 0 directory
all_images = [f for f in os.listdir(class_0_dir) if os.path.isfile(os.path.join(class_0_dir, f))]
print(f"Total images in class 0 before downsampling: {len(all_images)}")

# Only delete if there are more than target_count
if len(all_images) > target_count:
    # Randomly shuffle and keep only the first 1000
    to_keep = set(random.sample(all_images, target_count))

    for img in all_images:
        if img not in to_keep:
            os.remove(os.path.join(class_0_dir, img))

    print(f"Class 0 downsampled to {target_count} images.")
else:
    print("Class 0 already has less than or equal to 1000 images. No deletion performed.")





import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models, datasets

# --- Config ---
IMAGE_DIR = "/content/colored_images"
AUG_IMAGE_DIR = "/content/augmented_colored"
BATCH_SIZE = 32
NUM_CLASSES = 5  # 0 to 4 for DR severity
NUM_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model_path = "/content/drive/MyDrive/best_dr_model.pth"

# --- Data ---
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomResizedCrop(299, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Dataset ---
class DRDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label in os.listdir(image_dir):
            label_dir = os.path.join(image_dir, label)
            if os.path.isdir(label_dir):
                for image_name in os.listdir(label_dir):
                    self.image_paths.append(os.path.join(label_dir, image_name))
                    self.labels.append(int(label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label

# Load original and augmented datasets
orig_dataset = DRDataset(IMAGE_DIR, transform)
aug_dataset = DRDataset(AUG_IMAGE_DIR, transform)

# Combine datasets
full_dataset = ConcatDataset([orig_dataset, aug_dataset])

# Train/Val Split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# --- Model ---
model = models.inception_v3(pretrained=True, aux_logits=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# --- Loss, Optimizer, Scheduler ---
class_counts = [361, 74, 201, 37, 60]
class_weights = [1.0 / c for c in class_counts]
class_weights = torch.tensor(class_weights).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# --- Helper function to evaluate accuracy ---
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total

# --- Training Loop ---
best_val_accuracy = 0.0  # Initialize with 0 or a very low value

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs, aux_outputs = model(images)

        loss1 = criterion(outputs, labels)
        loss2 = criterion(aux_outputs, labels)
        loss = loss1 + 0.4 * loss2

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation after each epoch
    val_accuracy = evaluate_accuracy(model, val_loader, DEVICE)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {running_loss / len(train_loader):.4f} - Val Accuracy: {val_accuracy:.2f}%")

    # Save the model if it's the best so far
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model with accuracy {best_val_accuracy:.2f}% to {best_model_path}")

    scheduler.step()
# --- Evaluation ---
# Load the best model before final evaluation
model.load_state_dict(torch.load(best_model_path))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nClassification Report:\n", classification_report(all_labels, all_preds))






import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             mean_squared_error)
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
import torch.nn as nn
import numpy as np

# --- Configuration ---
IMAGE_DIR = "/content/colored_images"
AUG_IMAGE_DIR = "/content/augmented_colored"
BATCH_SIZE = 32
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model_path = "/content/drive/MyDrive/best_dr_model.pth"

# --- Transformations ---
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Dataset Class ---
class DRDataset(Dataset):
    def __init__(self, class_dirs, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label in class_dirs:
            label_dir = class_dirs[label]
            for fname in os.listdir(label_dir):
                if fname.endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(label_dir, fname))
                    self.labels.append(int(label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# --- Combine datasets: 0 and 2 from original, 1, 3, 4 from augmented ---
combined_dirs = {
    "0": os.path.join(IMAGE_DIR, "0"),
    "2": os.path.join(IMAGE_DIR, "2"),
    "1": os.path.join(AUG_IMAGE_DIR, "1"),
    "3": os.path.join(AUG_IMAGE_DIR, "3"),
    "4": os.path.join(AUG_IMAGE_DIR, "4"),
}

full_dataset = DRDataset(combined_dirs, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# --- Model Setup ---
model = models.inception_v3(pretrained=True, aux_logits=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# --- Load Best Model ---
model.load_state_dict(torch.load(best_model_path))
model.eval()

# --- Evaluation Metrics ---
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

conf_mat = confusion_matrix(all_labels, all_preds)
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
mse = mean_squared_error(all_labels, all_preds)
specificity = np.mean([conf_mat[i][i] / (sum(conf_mat[:, i]) + 1e-6) for i in range(NUM_CLASSES)])
fpr = np.mean([1 - conf_mat[i][i] / (sum(conf_mat[i]) + 1e-6) for i in range(NUM_CLASSES)])
fnr = np.mean([1 - conf_mat[i][i] / (sum(conf_mat[:, i]) + 1e-6) for i in range(NUM_CLASSES)])
auc = roc_auc_score(all_labels, np.eye(NUM_CLASSES)[all_preds], multi_class='ovo')

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall (Sensitivity)": recall,
    "F1 Score": f1,
    "Specificity": specificity,
    "False Positive Rate (FPR)": fpr,
    "False Negative Rate (FNR)": fnr,
    "AUC": auc,
    "MSE": mse,
    "Confusion Matrix": conf_mat.tolist()
}

metrics
