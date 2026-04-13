import json
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from torchvision import transforms
from model_arch import AgroYieldNet
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    with open('classes.txt', 'r') as f:
        lines = f.readlines()
        breed_names = [name.strip() for name in lines[0].split(',')]
        disease_names = [name.strip() for name in lines[1].split(',')]
except FileNotFoundError:
    breed_names, disease_names = [], []

with open('breed_info.json', 'r') as f:
    breed_info_data = json.load(f)
    breed_info_lookup = {k.lower(): v for k, v in breed_info_data.items()}

num_breeds = len(breed_names)
num_diseases = len(disease_names)

model = AgroYieldNet(num_breeds=num_breeds, num_diseases=num_diseases).to(device)
model.load_state_dict(torch.load('agro_multi_task.pth', map_location=device, weights_only=True))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            b_logits, d_logits = model(tensor)
            b_probs = F.softmax(b_logits, dim=1)
            d_probs = F.softmax(d_logits, dim=1)
            b_conf, b_idx = torch.max(b_probs, 1)
            d_conf, d_idx = torch.max(d_probs, 1)

        breed_key = breed_names[b_idx.item()]
        disease_key = disease_names[d_idx.item()]
        
        breed_details = breed_info_lookup.get(breed_key.lower(), {})

        return {
            "breed": breed_key,
            "breed_confidence": float(b_conf.item()),
            "disease": disease_key,
            "disease_confidence": float(d_conf.item()),
            "breed_details": breed_details,
            "status": "Success"
        }
    except Exception as e:
        return {"status": "Error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)