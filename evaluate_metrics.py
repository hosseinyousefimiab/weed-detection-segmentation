import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score
from albumentations import Compose, Normalize
import segmentation_models_pytorch as smp

RESIZE_HEIGHT = 1312
RESIZE_WIDTH = 1312

root_dir = "/cta/users/hossein.yousefimiab/suger/data/"
output_dir = "/cta/users/hossein.yousefimiab/suger/output/"
os.makedirs(output_dir, exist_ok=True)

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

all_rgb_files = sorted(os.listdir(os.path.join(root_dir, "rgb")))
train_size = int(0.7 * len(all_rgb_files))
val_size = int(0.15 * len(all_rgb_files))
test_filenames = all_rgb_files[train_size + val_size:]

test_transform = Compose([Normalize(mean=(0.485, 0.456, 0.406, 0.5), std=(0.229, 0.224, 0.225, 0.2))])
test_dataset = BeetWeedDataset(root_dir, test_filenames, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

model_path = os.path.join(output_dir, "best_model.pth")
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=4, classes=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

all_preds = []
all_labels = []

valid_classes = [0, 1, 2]  
with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.numpy().flatten()
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy().flatten()
        mask = np.isin(masks, valid_classes) & np.isin(preds, valid_classes)
        all_preds.extend(preds[mask])
        all_labels.extend(masks[mask])

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

kappa = cohen_kappa_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average=None, labels=valid_classes)
recall = recall_score(all_labels, all_preds, average=None, labels=valid_classes)

print(f"Cohen's ? Coefficient: {kappa:.4f}")
print(f"Precision (per class): {precision}")
print(f"Recall (per class): {recall}")

results_path = os.path.join(output_dir, "evaluation_metrics.txt")
with open(results_path, "w") as f:
    f.write(f"Cohen's ? Coefficient: {kappa:.4f}\n")
    f.write(f"Precision (per class): {precision}\n")
    f.write(f"Recall (per class): {recall}\n")
print(f"Results saved at {results_path}")
