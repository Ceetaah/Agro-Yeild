import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_arch import AgroYieldNet
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]) 

breed_dataset = datasets.ImageFolder(root='./dataset/breeds', transform=train_transform)
disease_dataset = datasets.ImageFolder(root='./dataset/diseases', transform=train_transform)

breed_loader = DataLoader(breed_dataset, batch_size=BATCH_SIZE, shuffle=True)
disease_loader = DataLoader(disease_dataset, batch_size=BATCH_SIZE, shuffle=True)

num_breeds = len(breed_dataset.classes)
num_diseases = len(disease_dataset.classes)

model = AgroYieldNet(num_breeds=num_breeds, num_diseases=num_diseases).to(device)

for param in model.backbone.parameters():
    param.requires_grad = False
for param in model.backbone.blocks[-2:].parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scaler = torch.amp.GradScaler('cuda')

for epoch in range(EPOCHS):
    model.train()
    breed_iter = iter(breed_loader)
    disease_iter = iter(disease_loader)
    
    steps = max(len(breed_loader), len(disease_loader))
    
    for i in range(steps):
        try:
            b_imgs, b_labels = next(breed_iter)
        except StopIteration:
            breed_iter = iter(breed_loader)
            b_imgs, b_labels = next(breed_iter)
            
        try:
            d_imgs, d_labels = next(disease_iter)
        except StopIteration:
            disease_iter = iter(disease_loader)
            d_imgs, d_labels = next(disease_iter)

        b_imgs, b_labels = b_imgs.to(device), b_labels.to(device)
        d_imgs, d_labels = d_imgs.to(device), d_labels.to(device)

        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            b_out, _ = model(b_imgs)
            _, d_out = model(d_imgs)
            
            loss_b = criterion(b_out, b_labels)
            loss_d = criterion(d_out, d_labels)
            loss = loss_b + loss_d

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.4f}")

torch.save(model.state_dict(), 'agro_multi_task.pth')
with open('classes.txt', 'w') as f:
    f.write(",".join(breed_dataset.classes) + "\n")
    f.write(",".join(disease_dataset.classes))