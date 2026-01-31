from fastapi import FastAPI, HTTPException, Body
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

app = FastAPI()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
labels_map = {0: "Clear", 1: "Crystals", 2: "Other", 3: "Precipitate"}

model = models.resnet34(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.fc.in_features, 1000),
    nn.ReLU(),
    nn.Linear(1000, 300),
    nn.ReLU(),
    nn.Linear(300, 4)
)

model.load_state_dict(torch.load("/data/brave/MARCO/MS/model_epoch_130.net", map_location=device))
model.to(device)
model.eval()

server_transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor()
])

@app.post("/classify")
async def classify_image(path: str = Body(..., embed=True)):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        image = Image.open(path).convert("RGB")
        
        input_tensor = server_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            conf, pred = torch.max(probs, 0)

        return {
            "status": "success",
            "prediction": labels_map[pred.item()],
            "confidence": round(float(conf), 4),
            "probabilities": {labels_map[i]: round(float(probs[i]), 4) for i in range(4)}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
