
import os
import json
import torch
import torch.nn as nn
import numpy as np

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Configuration
DATA_DIR = "dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
CLASS_PATH = os.path.join(MODEL_DIR, "class_names.json")
OUTPUT_PATH = os.path.join(MODEL_DIR, "evaluation_report.json")

BATCH_SIZE = 16
IMAGE_SIZE = 224
RANDOM_SEED = 42

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# Image preprocessing: same evaluation transform as train.py
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load dataset
print("\nLoading dataset...")

dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform
)

classes = dataset.classes
targets = np.array(dataset.targets)
indices = np.arange(len(targets))

# Load saved class names
with open(CLASS_PATH, "r") as file:
    saved_classes = json.load(file)

if classes != saved_classes:
    raise ValueError(
        "Dataset class order does not match saved model classes."
    )

print("Total images:", len(dataset))
print("Total classes:", len(classes))

# Recreate the exact original split from train.py
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

test_dataset = Subset(dataset, test_indices)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nTest images:", len(test_dataset))

# Load ResNet18 architecture
print("\nLoading trained ResNet18 model...")

model = models.resnet18(weights=None)

input_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(input_features, len(classes))
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.to(device)
model.eval()

print("Model loaded successfully!")

# Evaluate model
all_predictions = []
all_labels = []

print("\nEvaluating test dataset...")

with torch.inference_mode():

    for batch_index, (images, labels) in enumerate(test_loader):

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_labels.extend(labels.numpy().tolist())

        if (batch_index + 1) % 20 == 0:
            print(
                f"Processed batches: {batch_index + 1}"
            )

# Calculate metrics
accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision_macro = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

precision_weighted = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

# Class-wise report
report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(classes))),
    target_names=classes,
    output_dict=True,
    zero_division=0
)

# Confusion matrix
matrix = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(classes)))
)

# Create evaluation report
evaluation_results = {
    "model": "ResNet18",
    "dataset": "PlantVillage",
    "number_of_classes": len(classes),
    "test_images": len(test_dataset),
    "metrics": {
        "accuracy": round(float(accuracy) * 100, 2),
        "precision_macro": round(float(precision_macro) * 100, 2),
        "recall_macro": round(float(recall_macro) * 100, 2),
        "f1_macro": round(float(f1_macro) * 100, 2),
        "precision_weighted": round(float(precision_weighted) * 100, 2),
        "recall_weighted": round(float(recall_weighted) * 100, 2),
        "f1_weighted": round(float(f1_weighted) * 100, 2)
    },
    "class_names": classes,
    "class_wise_report": report,
    "confusion_matrix": matrix.tolist()
}

# Save report
with open(OUTPUT_PATH, "w") as file:
    json.dump(evaluation_results, file, indent=4)

# Display results
print("\n========== MODEL EVALUATION RESULTS ==========")

print(f"Accuracy:           {accuracy * 100:.2f}%")
print(f"Macro Precision:    {precision_macro * 100:.2f}%")
print(f"Macro Recall:       {recall_macro * 100:.2f}%")
print(f"Macro F1-score:     {f1_macro * 100:.2f}%")

print(f"Weighted Precision: {precision_weighted * 100:.2f}%")
print(f"Weighted Recall:    {recall_weighted * 100:.2f}%")
print(f"Weighted F1-score:  {f1_weighted * 100:.2f}%")

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(matrix)

print("\nEvaluation completed successfully!")
print("Report saved to:", OUTPUT_PATH)
