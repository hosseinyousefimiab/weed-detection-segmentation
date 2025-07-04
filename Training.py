import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import glob
import random
import matplotlib.pyplot as plt
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast, GaussianBlur, Normalize
)

RESIZE_HEIGHT = 1312
RESIZE_WIDTH = 1312

def get_corresponding_files(rgb_file):
    numeric_id = rgb_file.split("_")[1].split(".")[0]
    nir_file = rgb_file.replace("rgb", "nir")
    annotation_id = str(int(numeric_id))
    annotation_file = f"{annotation_id}.png"
    return nir_file, annotation_file

def color_mask_to_class_mask(mask_img):
    class_mask = np.zeros((mask_img.shape[0], mask_img.shape[1]), dtype=np.uint8)
    background = (mask_img[:, :, 0] == 0) & (mask_img[:, :, 1] == 0) & (mask_img[:, :, 2] == 0)
    sugarbeet = (mask_img[:, :, 2] == 255) & (mask_img[:, :, 1] == 0) & (mask_img[:, :, 0] == 0)
    weed = (mask_img[:, :, 0] == 255) & (mask_img[:, :, 1] == 0) & (mask_img[:, :, 2] == 0)
    class_mask[background] = 0
    class_mask[sugarbeet] = 1
    class_mask[weed] = 2
    return class_mask

class BeetWeedDataset(Dataset):
    def __init__(self, root_dir, image_names, transform=None):
        self.root_dir = root_dir
        self.image_names = self.filter_valid_files(image_names)
        self.transform = transform

    def filter_valid_files(self, image_names):
        valid_files = []
        for filename in image_names:
            nir_file, annotation_file = get_corresponding_files(filename)
            rgb_path = os.path.join(self.root_dir, "rgb", filename)
            nir_path = os.path.join(self.root_dir, "nir", nir_file)
            ann_path = os.path.join(self.root_dir, "annotations", annotation_file)
            if os.path.exists(rgb_path) and os.path.exists(nir_path) and os.path.exists(ann_path):
                valid_files.append(filename)
        return valid_files

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        filename = self.image_names[idx]
        nir_file, annotation_file = get_corresponding_files(filename)

        rgb_path = os.path.join(self.root_dir, "rgb", filename)
        nir_path = os.path.join(self.root_dir, "nir", nir_file)
        ann_path = os.path.join(self.root_dir, "annotations", annotation_file)

        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        nir = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
        ann = cv2.imread(ann_path, cv2.IMREAD_COLOR)

        # Resize images and masks
        rgb = cv2.resize(rgb, (RESIZE_WIDTH, RESIZE_HEIGHT))
        nir = cv2.resize(nir, (RESIZE_WIDTH, RESIZE_HEIGHT))
        ann = cv2.resize(ann, (RESIZE_WIDTH, RESIZE_HEIGHT))

        ann = color_mask_to_class_mask(ann)
        nir = np.expand_dims(nir, axis=-1)
        input_img = np.concatenate((rgb, nir), axis=-1)

        if self.transform is not None:
            augmented = self.transform(image=input_img, mask=ann)
            input_img, ann = augmented["image"], augmented["mask"]

        input_img = torch.from_numpy(np.transpose(input_img, (2, 0, 1))).float()
        ann = torch.from_numpy(ann).long()

        return input_img, ann

def calculate_metrics(pred, target, num_classes=3):
    iou_scores, dice_scores = [], []
    for cls in range(num_classes):
        intersection = ((pred == cls) & (target == cls)).sum()
        union = ((pred == cls) | (target == cls)).sum()
        dice = (2 * intersection) / (union + intersection + 1e-7)
        iou = intersection / (union + 1e-7)
        iou_scores.append(iou)
        dice_scores.append(dice)
    return iou_scores, dice_scores

root_dir = "/cta/users/hossein.yousefimiab/suger/data/"
output_dir = "/cta/users/hossein.yousefimiab/suger/output/"
os.makedirs(output_dir, exist_ok=True)

all_rgb_files = glob.glob(os.path.join(root_dir, "rgb", "*.png"))
all_filenames = [os.path.basename(p) for p in all_rgb_files]
random.shuffle(all_filenames)

train_size = int(0.7 * len(all_filenames))
val_size = int(0.15 * len(all_filenames))

train_filenames = all_filenames[:train_size]
val_filenames = all_filenames[train_size:train_size + val_size]

train_transform = Compose([
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.2),
    GaussianBlur(p=0.2),
    Normalize(mean=(0.485, 0.456, 0.406, 0.5), std=(0.229, 0.224, 0.225, 0.2)),
])

val_transform = Compose([Normalize(mean=(0.485, 0.456, 0.406, 0.5), std=(0.229, 0.224, 0.225, 0.2))])

train_dataset = BeetWeedDataset(root_dir, train_filenames, transform=train_transform)
val_dataset = BeetWeedDataset(root_dir, val_filenames, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=4, classes=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 20
best_val_loss = float("inf")
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    iou, dice = [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            masks = masks.cpu().numpy()
            iou_epoch, dice_epoch = calculate_metrics(preds, masks)
            iou.append(iou_epoch)
            dice.append(dice_epoch)
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    avg_iou = np.mean(iou, axis=0)
    avg_dice = np.mean(dice, axis=0)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"Avg IoU: {avg_iou}, Avg Dice: {avg_dice}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
        print("Best model saved.")

plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Convergence")
plt.legend()
plt.savefig(os.path.join(output_dir, "convergence_graph.png"))
plt.close()
print(f"Convergence graph saved at {os.path.join(output_dir, 'convergence_graph.png')}")
