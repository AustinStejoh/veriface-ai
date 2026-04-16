import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os, random, time

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
DATA_DIR   = "data/real_vs_fake/real-vs-fake"
SAVE_PATH  = "models/best_model.pth"
BATCH_SIZE = 16
EPOCHS     = 5
LR         = 0.0003
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_SIZE = 18000
VAL_SIZE   = 2000

print("=" * 50)
print(f"  Device  : {DEVICE}")
print(f"  Train   : {TRAIN_SIZE} images")
print(f"  Val     : {VAL_SIZE} images")
print(f"  Epochs  : {EPOCHS}")
print("=" * 50)

# ─────────────────────────────────────────
#  DATA TRANSFORMS
# ─────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────
print("\nLoading dataset...")

full_train = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"), train_transform
)
full_val = datasets.ImageFolder(
    os.path.join(DATA_DIR, "valid"), val_transform
)

# Random subset for CPU
train_indices = random.sample(range(len(full_train)), TRAIN_SIZE)
val_indices   = random.sample(range(len(full_val)),   VAL_SIZE)

train_ds = Subset(full_train, train_indices)
val_ds   = Subset(full_val,   val_indices)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)

print(f"  Classes : {full_train.classes}")
print(f"  Train   : {len(train_ds)} images ✅")
print(f"  Val     : {len(val_ds)} images ✅")

# ─────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────
print("\nLoading EfficientNet model...")
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features, 2
)
model = model.to(DEVICE)
if os.path.exists(SAVE_PATH):
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    print(f"  Loaded existing checkpoint: {SAVE_PATH}")
print("  Model ready ✅")

# ─────────────────────────────────────────
#  LOSS + OPTIMIZER
# ─────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=2, gamma=0.5
)

# ─────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────
best_val_acc = 0.0

print("\nStarting training...\n")

for epoch in range(EPOCHS):
    start = time.time()

    # ── Train ──
    model.train()
    train_loss, train_correct = 0, 0

    for images, labels in tqdm(train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

    # ── Validate ──
    model.eval()
    val_loss, val_correct = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader,
                desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  "):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            val_loss    += loss.item()
            val_correct += (outputs.argmax(1) == labels).sum().item()

    # ── Stats ──
    train_acc = train_correct / len(train_ds) * 100
    val_acc   = val_correct   / len(val_ds)   * 100
    elapsed   = (time.time() - start) / 60

    print(f"\n📊 Epoch {epoch+1}/{EPOCHS}  ({elapsed:.1f} min)")
    print(f"   Train → Acc: {train_acc:.2f}%  Loss: {train_loss/len(train_loader):.4f}")
    print(f"   Val   → Acc: {val_acc:.2f}%  Loss: {val_loss/len(val_loader):.4f}")

    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"   💾 Best model saved! ({val_acc:.2f}%)")

print(f"\n{'='*50}")
print(f"  Training Complete!")
print(f"  Best Val Accuracy : {best_val_acc:.2f}%")
print(f"  Model saved at    : {SAVE_PATH}")
print(f"{'='*50}")