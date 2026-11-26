
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

# -------------------------------
# 1. Configuration
# -------------------------------

DATA_DIR = "dataset"
MODEL_DIR = "models"

BATCH_SIZE = 5
EPOCHS = 3
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
RANDOM_SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 2. Image transformations
# -------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# 3. Load dataset
# -------------------------------

print("\nLoading dataset...")

full_dataset = datasets.ImageFolder(DATA_DIR)

classes = full_dataset.classes
targets = np.array(full_dataset.targets)
indices = np.arange(len(targets))

print("Total images:", len(full_dataset))
print("Total classes:", len(classes))

for index, class_name in enumerate(classes):
    print(class_name, ":", np.sum(targets == index))

# -------------------------------
# 4. Train / Validation / Test split
# -------------------------------

train_indices, temp_indices = train_test_split(
    indices,
    test_size=0.30,
    stratify=targets,
    random_state=RANDOM_SEED
)

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    stratify=targets[temp_indices],
    random_state=RANDOM_SEED
)

print("\nDataset split:")
print("Training:", len(train_indices))
print("Validation:", len(val_indices))
print("Testing:", len(test_indices))

# Separate datasets with correct transforms
train_dataset = datasets.ImageFolder(
    DATA_DIR, transform=train_transform
)

eval_dataset = datasets.ImageFolder(
    DATA_DIR, transform=test_transform
)

train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(eval_dataset, val_indices)
test_subset = Subset(eval_dataset, test_indices)

# -------------------------------
# 5. Handle class imbalance
# -------------------------------

train_labels = targets[train_indices]
class_counts = np.bincount(train_labels)

class_weights = 1.0 / class_counts
sample_weights = class_weights[train_labels]

sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# -------------------------------
# 6. Load ResNet18
# -------------------------------

print("\nLoading ResNet18 model...")

try:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
except Exception as error:
    print("Pretrained weights unavailable:", error)
    print("Using ResNet18 without pretrained weights.")
    model = models.resnet18(weights=None)

# Freeze feature extraction layers
for parameter in model.parameters():
    parameter.requires_grad = False

# Replace final classification layer
input_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(input_features, len(classes))
)

model = model.to(device)

# -------------------------------
# 7. Loss and optimizer
# -------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)

# -------------------------------
# 8. Training function
# -------------------------------

def train_one_epoch():
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy

# -------------------------------
# 9. Validation function
# -------------------------------

def validate():
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, 100 * correct / total

# -------------------------------
# 10. Model training
# -------------------------------

print("\nStarting training...\n")

best_accuracy = 0.0
history = []

for epoch in range(EPOCHS):

    train_loss, train_accuracy = train_one_epoch()

    val_loss, val_accuracy = validate()

    scheduler.step()

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            os.path.join(MODEL_DIR, "best_model.pth")
        )

        print("Best model saved!")

# -------------------------------
# 11. Save class names and history
# -------------------------------

with open(
    os.path.join(MODEL_DIR, "class_names.json"),
    "w"
) as file:
    json.dump(classes, file, indent=4)

with open(
    os.path.join(MODEL_DIR, "training_history.json"),
    "w"
) as file:
    json.dump(history, file, indent=4)

print("\nTraining completed!")
print("Best validation accuracy:", round(best_accuracy, 2), "%")

# -------------------------------
# 12. Test the best model
# -------------------------------

print("\nEvaluating on test dataset...")

model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, "best_model.pth"),
        map_location=device,
        weights_only=True
    )
)

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=classes,
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_predictions))

print("\nProject model training and evaluation completed!")
