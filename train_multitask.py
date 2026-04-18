import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_arch import AgroYieldNet
import os

a1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a2 = 32
a3 = 20
a4 = 1e-4

a5 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]) 

a6 = datasets.ImageFolder(root='./dataset/breeds', transform=a5)
a7 = datasets.ImageFolder(root='./dataset/diseases', transform=a5)

a8 = DataLoader(a6, batch_size=a2, shuffle=True)
a9 = DataLoader(a7, batch_size=a2, shuffle=True)

a10 = len(a6.classes)
a11 = len(a7.classes)

a12 = AgroYieldNet(num_breeds=a10, num_diseases=a11).to(a1)

for b1 in a12.backbone.parameters():
    b1.requires_grad = False
for b2 in a12.backbone.blocks[-2:].parameters():
    b2.requires_grad = True

a13 = nn.CrossEntropyLoss()
a14 = torch.optim.AdamW(filter(lambda b3: b3.requires_grad, a12.parameters()), lr=a4)
a15 = torch.amp.GradScaler('cuda')

for b4 in range(a3):
    a12.train()
    b5 = iter(a8)
    b6 = iter(a9)
    
    b7 = max(len(a8), len(a9))
    
    for b8 in range(b7):
        try:
            b9, b10 = next(b5)
        except StopIteration:
            b5 = iter(a8)
            b9, b10 = next(b5)
            
        try:
            b11, b12 = next(b6)
        except StopIteration:
            b6 = iter(a9)
            b11, b12 = next(b6)

        b9, b10 = b9.to(a1), b10.to(a1)
        b11, b12 = b11.to(a1), b12.to(a1)

        a14.zero_grad()
        
        with torch.amp.autocast('cuda'):
            c1, _ = a12(b9)
            _, c2 = a12(b11)
            
            c3 = a13(c1, b10)
            c4 = a13(c2, b12)
            c5 = c3 + c4

        a15.scale(c5).backward()
        a15.step(a14)
        a15.update()

    print(f"Epoch {b4+1}/{a3} - Loss: {c5.item():.4f}")

torch.save(a12.state_dict(), 'agro_multi_task.pth')
with open('classes.txt', 'w') as b13:
    b13.write(",".join(a6.classes) + "\n")
    b13.write(",".join(a7.classes))
