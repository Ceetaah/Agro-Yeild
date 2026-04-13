import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report
from model_arch import AgroYieldNet

# 1. Flexible Dataset for single-task evaluation
class AgroEvalDataset(Dataset):
    def __init__(self, root_dir, class_list, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {name.strip().lower(): i for i, name in enumerate(class_list)}
        self.images = []

        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                clean_name = folder_name.strip().lower()
                if clean_name in self.class_to_idx:
                    idx = self.class_to_idx[clean_name]
                    for img_name in os.listdir(folder_path):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.images.append((os.path.join(folder_path, img_name), idx))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path, label = self.images[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def run_evaluation(model_path, classes_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(classes_path, 'r') as f:
        lines = f.readlines()
        breed_names = [n.strip() for n in lines[0].split(',')]
        disease_names = [n.strip() for n in lines[1].split(',')]

    model = AgroYieldNet(num_breeds=len(breed_names), num_diseases=len(disease_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- EVALUATE BREEDS ---
    print("\n--- Testing Breed Identification ---")
    breed_ds = AgroEvalDataset('dataset/breeds', breed_names, transform=eval_transforms)
    if len(breed_ds) > 0:
        loader = DataLoader(breed_ds, batch_size=32, shuffle=False)
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                b_out, _ = model(imgs.to(device))
                y_pred.extend(torch.max(b_out, 1)[1].cpu().numpy())
                y_true.extend(labels.numpy())
        print(classification_report(y_true, y_pred, target_names=breed_names, labels=range(len(breed_names)), zero_division=0))
    else:
        print("No breed images found in dataset/breeds")

    # --- EVALUATE DISEASES ---
    print("\n--- Testing Disease Detection ---")
    disease_ds = AgroEvalDataset('dataset/diseases', disease_names, transform=eval_transforms)
    if len(disease_ds) > 0:
        loader = DataLoader(disease_ds, batch_size=32, shuffle=False)
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                _, d_out = model(imgs.to(device))
                y_pred.extend(torch.max(d_out, 1)[1].cpu().numpy())
                y_true.extend(labels.numpy())
        print(classification_report(y_true, y_pred, target_names=disease_names, labels=range(len(disease_names)), zero_division=0))
    else:
        print("No disease images found in dataset/diseases")

if __name__ == "__main__":
    run_evaluation('agro_multi_task.pth', 'classes.txt')