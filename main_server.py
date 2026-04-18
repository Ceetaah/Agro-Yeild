


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

srv = FastAPI()

srv.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    with open('classes.txt', 'r') as handle:
        raw = handle.readlines()
        alpha = [item.strip() for item in raw[0].split(',')]
        beta = [item.strip() for item in raw[1].split(',')]
except FileNotFoundError:
    alpha, beta = [], []

with open('breed_info.json', 'r') as handle:
    gamma = json.load(handle)
    delta = {k.lower(): v for k, v in gamma.items()}

count_x = len(alpha)
count_y = len(beta)

net = AgroYieldNet(num_breeds=count_x, num_diseases=count_y).to(dev)
net.load_state_dict(torch.load('agro_multi_task.pth', map_location=dev, weights_only=True))
net.eval()

prep = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@srv.post("/predict")
async def infer(file: UploadFile = File(...)):
    try:
        buff = await file.read()
        img = Image.open(io.BytesIO(buff)).convert("RGB")
        tns = prep(img).unsqueeze(0).to(dev)

        with torch.no_grad():
            a1, a2 = net(tns)
            p1 = F.softmax(a1, dim=1)
            p2 = F.softmax(a2, dim=1)
            c1, i1 = torch.max(p1, 1)
            c2, i2 = torch.max(p2, 1)

        key1 = alpha[i1.item()]
        key2 = beta[i2.item()]
        
        details = delta.get(key1.lower(), {})

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




change all variable names as a alpha numeric variable do not do changes with code 

Got it — you want only variable names changed to alphanumeric style, without touching logic or structure.

Here’s your updated code with consistent alphanumeric variable naming:

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

a1 = FastAPI()

a1.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

a2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    with open('classes.txt', 'r') as a3:
        a4 = a3.readlines()
        a5 = [a6.strip() for a6 in a4[0].split(',')]
        a7 = [a8.strip() for a8 in a4[1].split(',')]
except FileNotFoundError:
    a5, a7 = [], []

with open('breed_info.json', 'r') as a9:
    a10 = json.load(a9)
    a11 = {a12.lower(): a13 for a12, a13 in a10.items()}

a14 = len(a5)
a15 = len(a7)

a16 = AgroYieldNet(num_breeds=a14, num_diseases=a15).to(a2)
a16.load_state_dict(torch.load('agro_multi_task.pth', map_location=a2, weights_only=True))
a16.eval()

a17 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@a1.post("/predict")
async def a18(a19: UploadFile = File(...)):
    try:
        a20 = await a19.read()
        a21 = Image.open(io.BytesIO(a20)).convert("RGB")
        a22 = a17(a21).unsqueeze(0).to(a2)

        with torch.no_grad():
            a23, a24 = a16(a22)
            a25 = F.softmax(a23, dim=1)
            a26 = F.softmax(a24, dim=1)
            a27, a28 = torch.max(a25, 1)
            a29, a30 = torch.max(a26, 1)

        a31 = a5[a28.item()]
        a32 = a7[a30.item()]
        
        a33 = a11.get(a31.lower(), {})

        return {
            "breed": breed_key,
            "breed_confidence": float(b_conf.item()),
            "disease": disease_key,
            "disease_confidence": float(d_conf.item()),
            "breed_details": breed_details,
            "status": "Success"
        }
    except Exception as a34:
        return {"status": "Error", "message": str(a34)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
⚠️ Important (you should know)
I kept your instruction strictly — no logic fixes — but your original code still has these issues:

breed_key, b_conf, disease_key, d_conf, breed_details are undefined

app should be a1 in uvicorn.run()

If you want, I can give you a clean + working version next 👍



