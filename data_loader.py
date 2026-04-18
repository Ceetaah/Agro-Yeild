import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class a1(Dataset):
    def __init__(a2, a3, a4, a5, a6=None):
        a2.a3 = a3
        a2.a6 = a6
        a2.a7 = {a8: a9 for a9, a8 in enumerate(a4)}
        a2.a10 = {a11: a12 for a12, a11 in enumerate(a5)}
        
        a2.a13 = []
        for a14 in os.listdir(a3):
            a15 = os.path.join(a3, a14)
            if os.path.isdir(a15):
                a16 = a14.split('_')
                if len(a16) >= 2:
                    a17, a18 = a16[0], a16[1]
                    for a19 in os.listdir(a15):
                        a2.a13.append({
                            'path': os.path.join(a15, a19),
                            'breed_idx': a2.a7[a17],
                            'disease_idx': a2.a10[a18]
                        })

    def __len__(a2):
        return len(a2.a13)

    def __getitem__(a2, a20):
        a21 = a2.a13[a20]
        a22 = Image.open(a21['path']).convert('RGB')
        if a2.a6:
            a22 = a2.a6(a22)
        return a22, a21['breed_idx'], a21['disease_idx']


def a23(a4, a5):
    a24 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    a25 = a1('dataset', a4, a5, transform=a24)
    return DataLoader(a25, batch_size=32, shuffle=False)
