import torch
import torch.nn as nn
import timm

class AgroYieldNet(nn.Module):
    def __init__(self, num_breeds, num_diseases):
        super(AgroYieldNet, self).__init__()
        
        self.backbone = timm.create_model('vit_base_patch16_dinov3', pretrained=True)
        self.feature_dim = self.backbone.num_features
        
        self.breed_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_breeds)
        )
        
        self.disease_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_diseases)
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)
        cls_token = features[:, 0]
        
        breed_logits = self.breed_head(cls_token)
        disease_logits = self.disease_head(cls_token)
        
        return breed_logits, disease_logits