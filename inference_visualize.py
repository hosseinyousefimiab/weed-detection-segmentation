import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import glob
import random
from albumentations import Compose, Normalize
import matplotlib.pyplot as plt

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

def pad_to_1312(img):
    return cv2.resize(img, (1312, 1312), interpolation=cv2.INTER_LINEAR)

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
        rgb, nir, ann = pad_to_1312(rgb), pad_to_1312(nir), pad_to_1312(ann)
        ann = color_mask_to_class_mask(ann)
        nir = np.expand_dims(nir, axis=-1)
        input_img = np.concatenate((rgb, nir), axis=-1)
        if self.transform is not None:
            augmented = self.transform(image=input_img, mask=ann)
            input_img, ann = augmented["image"], augmented["mask"]
        input_img = torch.from_numpy(input_img.transpose(2, 0, 1)).float()
        ann = torch.from_numpy(ann).long()
        return input_img, ann

root_dir = "/cta/users/hossein.yousefimiab/suger/data/"
model_path = "/cta/users/hossein.yousefimiab/suger/output/best_model.pth"
output_dir = "/cta/users/hossein.yousefimiab/suger/output/"  

os.makedirs(output_dir, exist_ok=True)

all_rgb_files = glob.glob(os.path.join(root_dir, "rgb", "*.png"))
all_filenames = [os.path.basename(p) for p in all_rgb_files]
random.shuffle(all_filenames)
test_filenames = all_filenames[int(0.85 * len(all_filenames)):]

val_transform = Compose([Normalize(mean=(0.485, 0.456, 0.406, 0.5), std=(0.229, 0.224, 0.225, 0.2))])
test_dataset = BeetWeedDataset(root_dir, test_filenames, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=4, classes=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def save_predictions(model, dataloader, output_dir, num_samples=10):
    model.eval()
    saved_samples = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            masks = masks.cpu().numpy()

            for i in range(images.shape[0]):
                if saved_samples >= num_samples:
                    break

                rgb_image = images[i][:3].cpu().numpy().transpose(1, 2, 0)
                rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(rgb_image)
                axs[0].set_title("Input Image")
                axs[0].axis("off")

                axs[1].imshow(masks[i], cmap="viridis")
                axs[1].set_title("Ground Truth Mask")
                axs[1].axis("off")

                axs[2].imshow(preds[i], cmap="viridis")
                axs[2].set_title("Predicted Mask")
                axs[2].axis("off")

                plt.tight_layout()
                output_file = os.path.join(output_dir, f"prediction_{saved_samples + 1}.png")
                plt.savefig(output_file)
                plt.close()
                saved_samples += 1

            if saved_samples >= num_samples:
                break

def calculate_iou(preds, masks, num_classes=3):
    iou_scores = []
    for cls in range(num_classes):
        intersection = (preds == cls) & (masks == cls)
        union = (preds == cls) | (masks == cls)
        iou = (intersection.sum()) / (union.sum() + 1e-7)
        iou_scores.append(iou)
    return iou_scores

def dice_coefficient(pred, target, num_classes=3):
    dice_scores = []
    for cls in range(num_classes):
        intersection = ((pred == cls) & (target == cls)).sum()
        union = ((pred == cls) | (target == cls)).sum()
        dice = (2 * intersection) / (union + intersection + 1e-7)
        dice_scores.append(dice)
    return dice_scores

iou_scores, dice_scores = [], []
with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        masks = masks.cpu().numpy()
        iou_scores.append(calculate_iou(preds, masks))
        dice_scores.append(dice_coefficient(preds, masks))

avg_iou = np.mean(iou_scores, axis=0)
avg_dice = np.mean(dice_scores, axis=0)

plt.figure(figsize=(10, 5))
plt.bar(range(len(avg_iou)), avg_iou, tick_label=["Background", "Sugarbeet", "Weed"])
plt.title("IoU Scores for Each Class")
plt.ylabel("IoU")
plt.xlabel("Class")
iou_graph_path = os.path.join(output_dir, "iou_scores.png")
plt.savefig(iou_graph_path)
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(range(len(avg_dice)), avg_dice, tick_label=["Background", "Sugarbeet", "Weed"])
plt.title("Dice Coefficients for Each Class")
plt.ylabel("Dice Coefficient")
plt.xlabel("Class")
dice_graph_path = os.path.join(output_dir, "dice_coefficients.png")
plt.savefig(dice_graph_path)
plt.close()

save_predictions(model, test_loader, output_dir, num_samples=10)

print(f"Predictions and evaluation graphs saved in {output_dir}")
