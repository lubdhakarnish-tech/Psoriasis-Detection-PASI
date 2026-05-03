# ================================
# 1. IMPORTS
# ================================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from PIL import Image
import random
from collections import defaultdict

# ================================
# 2. PATH
# ================================
train_dir = "dataset/train"
test_dir = "dataset/test"

# ================================
# 3. TRANSFORMS
# ================================
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================================
# 4. LOAD DATASET
# ================================
train_dataset_full = datasets.ImageFolder(train_dir, transform=train_transform)
test_dataset_full  = datasets.ImageFolder(test_dir, transform=test_transform)

print("Classes:", train_dataset_full.classes)

# ================================
# 5. BALANCE DATASET (FIXED)
# ================================
class_indices = defaultdict(list)

for idx, (_, label) in enumerate(train_dataset_full):
    class_indices[label].append(idx)

print("\nClass Distribution:")
for label in class_indices:
    print(f"Class {label}: {len(class_indices[label])} images")

# FIX HERE
min_class = min(len(v) for v in class_indices.values())
samples_per_class = min(min_class, 300)

print("Using samples per class:", samples_per_class)

balanced_indices = []
for label in class_indices:
    balanced_indices += random.sample(class_indices[label], samples_per_class)

random.shuffle(balanced_indices)

train_dataset = Subset(train_dataset_full, balanced_indices)
test_dataset = Subset(test_dataset_full, range(200))
# ================================
# 6. DATALOADER
# ================================
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=0)

# ================================
# 7. MODEL (FREEZE BACKBONE)
# ================================
model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ================================
# 8. LOSS + OPTIMIZER
# ================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0005)

# ================================
# 9. TRAINING
# ================================
epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    total_loss /= len(train_loader.dataset)

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")  # ✔ inside

# ================================
# 10. TEST ACCURACY
# ================================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print("\nTest Accuracy:", accuracy, "%")

# ================================
# 11. TEST SINGLE IMAGE
# ================================
img_path = "C:/Users/KIIT0001/Documents/dataset/test/Psoriasis/gg.jpeg"

img = Image.open(img_path).convert("RGB")
img = test_transform(img).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    output = model(img)
    pred = torch.argmax(output)

classes = train_dataset_full.classes
print("Prediction:", classes[pred])

# ================================
# 12. PASI SCORING (ADDED)
# ================================
import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_lesion(img_np):
    """
    Simple baseline segmentation using LAB color space (A channel)
    """
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    A_blur = cv2.GaussianBlur(A, (5,5), 0)

    # Otsu threshold
    _, mask = cv2.threshold(A_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleaning
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    final_mask = np.zeros_like(mask)

    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        final_mask[labels == largest] = 1

    return final_mask


def compute_pasi(mask, img):
    """
    Basic PASI approximation
    """
    R = img[:,:,0]
    G = img[:,:,1]

    # Area (fraction of lesion)
    area = np.sum(mask) / mask.size

    # Erythema (redness)
    erythema = np.mean((R - G)[mask > 0]) if np.sum(mask) > 0 else 0

    # Scaling (texture variation)
    scaling = np.std(img[mask > 0]) if np.sum(mask) > 0 else 0

    # Thickness (edge intensity)
    edges = cv2.Canny(img, 100, 200)
    thickness = np.mean(edges[mask > 0]) if np.sum(mask) > 0 else 0

    # Normalize to PASI-like scale (0–4)
    ery = min(4, erythema / 50)
    scl = min(4, scaling / 50)
    thk = min(4, thickness / 50)

    pasi = area * (ery + scl + thk)

    return pasi, area, ery, scl, thk


# ================================
# 13. RUN PASI ONLY IF PSORIASIS
# ================================
if classes[pred] == "Psoriasis":

    print("\n--- Running PASI Scoring ---")

    # Convert PIL → numpy
    img_np = np.array(Image.open(img_path).convert("RGB"))

    # Segmentation
    mask = segment_lesion(img_np)

    # PASI calculation
    pasi, area, ery, scl, thk = compute_pasi(mask, img_np)

    print(f"Area: {area:.3f}")
    print(f"Erythema: {ery:.3f}")
    print(f"Scaling: {scl:.3f}")
    print(f"Thickness: {thk:.3f}")
    print(f"Final PASI Score: {pasi:.3f}")

    # ================================
    # 14. VISUALIZATION
    # ================================
    overlay = img_np.copy()
    overlay[mask > 0] = [255, 0, 0]

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(img_np)
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis('off')

    plt.show()

else:
    print("\nNo PASI calculation (Normal skin detected)")
