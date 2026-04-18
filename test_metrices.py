import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report
from model_arch import AgroYieldNet

class a1(Dataset):
    def __init__(a2, a3, a4, a5=None):
        a2.a3 = a3
        a2.a5 = a5
        a2.a6 = {a7.strip().lower(): a8 for a8, a7 in enumerate(a4)}
        a2.a9 = []

        for a10 in os.listdir(a3):
            a11 = os.path.join(a3, a10)
            if os.path.isdir(a11):
                a12 = a10.strip().lower()
                if a12 in a2.a6:
                    a13 = a2.a6[a12]
                    for a14 in os.listdir(a11):
                        if a14.lower().endswith(('.png', '.jpg', '.jpeg')):
                            a2.a9.append((os.path.join(a11, a14), a13))

    def __len__(a2):
        return len(a2.a9)

    def __getitem__(a2, a15):
        a16, a17 = a2.a9[a15]
        a18 = Image.open(a16).convert('RGB')
        if a2.a5:
            a18 = a2.a5(a18)
        return a18, a17

def a19(a20, a21):
    a22 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(a21, 'r') as a23:
        a24 = a23.readlines()
        a25 = [a26.strip() for a26 in a24[0].split(',')]
        a27 = [a28.strip() for a28 in a24[1].split(',')]

    a29 = AgroYieldNet(num_breeds=len(a25), num_diseases=len(a27)).to(a22)
    a29.load_state_dict(torch.load(a20, map_location=a22, weights_only=True))
    a29.eval()

    a30 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("\n--- Testing Breed Identification ---")
    a31 = a1('dataset/breeds', a25, transform=a30)
    if len(a31) > 0:
        a32 = DataLoader(a31, batch_size=32, shuffle=False)
        a33, a34 = [], []
        with torch.no_grad():
            for a35, a36 in a32:
                a37, _ = a29(a35.to(a22))
                a34.extend(torch.max(a37, 1)[1].cpu().numpy())
                a33.extend(a36.numpy())
        print(classification_report(a33, a34, target_names=a25, labels=range(len(a25)), zero_division=0))
    else:
        print("No breed images found in dataset/breeds")

    print("\n--- Testing Disease Detection ---")
    a38 = a1('dataset/diseases', a27, transform=a30)
    if len(a38) > 0:
        a39 = DataLoader(a38, batch_size=32, shuffle=False)
        a40, a41 = [], []
        with torch.no_grad():
            for a42, a43 in a39:
                _, a44 = a29(a42.to(a22))
                a41.extend(torch.max(a44, 1)[1].cpu().numpy())
                a40.extend(a43.numpy())
        print(classification_report(a40, a41, target_names=a27, labels=range(len(a27)), zero_division=0))
    else:
        print("No disease images found in dataset/diseases")

if __name__ == "__main__":
    a19('agro_multi_task.pth', 'classes.txt')
