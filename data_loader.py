import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class AgroTestDataset(Dataset):
    def __init__(self, root_dir, breed_list, disease_list, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.breed_to_idx = {name: i for i, name in enumerate(breed_list)}
        self.disease_to_idx = {name: i for i, name in enumerate(disease_list)}
        
        self.images = []
        # Walk through folders: dataset/Breed_Disease/image.jpg
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                # Split folder name (e.g., "Ayrshire_Healthy" -> ["Ayrshire", "Healthy"])
                parts = folder_name.split('_')
                if len(parts) >= 2:
                    b_name, d_name = parts[0], parts[1]
                    for img_name in os.listdir(folder_path):
                        self.images.append({
                            'path': os.path.join(folder_path, img_name),
                            'breed_idx': self.breed_to_idx[b_name],
                            'disease_idx': self.disease_to_idx[d_name]
                        })

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        image = Image.open(img_info['path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_info['breed_idx'], img_info['disease_idx']

# Updated Loader Function
def get_test_loader(breed_list, disease_list):
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Use 'dataset' as the root since your 'test' folder wasn't found
    dataset = AgroTestDataset('dataset', breed_list, disease_list, transform=test_transforms)
    return DataLoader(dataset, batch_size=32, shuffle=False)