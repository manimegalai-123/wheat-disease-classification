import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import time

# CONFIG
DATA_DIR = 'wheat_leaf'  # Change this path to your dataset root
BATCH_SIZE = 32
NUM_CLASSES = len(os.listdir(DATA_DIR))  # Each folder is a class
EPOCHS = 10
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA TRANSFORMS
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# DATA LOADERS
train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR), transform=train_transforms)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# MODEL: EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# LOSS + OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# TRAINING LOOP
def train():
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(DEVICE), val_labels.to(DEVICE)
                val_outputs = model(val_images)
                _, val_preds = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_preds == val_labels).sum().item()
        val_acc = 100 * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}%\n")

# =========================
# START TRAINING
# =========================
if __name__ == "__main__":
    start = time.time()
    train()
    end = time.time()
    print(f"Training completed in {(end - start):.2f} seconds")

    # Save the model
    torch.save(model.state_dict(), 'wheat_leaf_model_efficientnet.pth')
    print("Model saved as wheat_leaf_model_efficientnet.pth")
