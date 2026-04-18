import torch
import torch.nn as nn
import timm

class a1(nn.Module):
    def __init__(b1, b2, b3):
        super(a1, b1).__init__()
        
        b1.b4 = timm.create_model('vit_base_patch16_dinov3', pretrained=True)
        b1.b5 = b1.b4.num_features
        
        b1.b6 = nn.Sequential(
            nn.Linear(b1.b5, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, b2)
        )
        
        b1.b7 = nn.Sequential(
            nn.Linear(b1.b5, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, b3)
        )

    def forward(c1, c2):
        c3 = c1.b4.forward_features(c2)
        c4 = c3[:, 0]
        
        c5 = c1.b6(c4)
        c6 = c1.b7(c4)
        
        return c5, c6
