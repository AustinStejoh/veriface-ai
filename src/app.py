from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import base64
import io
import cv2
import os
import gdown

# ─────────────────────────────────────────
#  APP SETUP
# ─────────────────────────────────────────
app = FastAPI(
    title="Deepfake Detector API",
    description="Detects if an image is real or AI-generated",
    version="1.0.0"
)

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/best_model.pth"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    print("Downloading model from Google Drive...")
    gdown.download(
        "https://drive.google.com/uc?id=1WHgMaLV_HW0iRUl96ll50v2rZI0rwkm7",
        MODEL_PATH,
        quiet=False
    )
    print("Model downloaded ✅")

print("Loading model...")
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features, 2
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
model.to(DEVICE)
print("Model loaded ✅")

# ─────────────────────────────────────────
#  IMAGE TRANSFORM
# ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Class labels
CLASSES = ["fake", "real"]

# ─────────────────────────────────────────
#  GRAD-CAM HELPER
# ─────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model    = model
        self.gradient = None
        self.activation = None

        # Hook into last conv layer of EfficientNet
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activation = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        # Forward pass
        output = self.model(input_tensor)
        self.model.zero_grad()

        # Backward pass for target class
        target = output[0][class_idx]
        target.backward()

        # Generate heatmap
        weights  = self.gradient.mean(dim=(2, 3), keepdim=True)
        cam      = (weights * self.activation).sum(dim=1, keepdim=True)
        cam      = torch.relu(cam)
        cam      = cam.squeeze().cpu().numpy()

        # Normalize
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()

        return cam

gradcam = GradCAM(model)

# ─────────────────────────────────────────
#  HELPER — overlay heatmap on image
# ─────────────────────────────────────────
def apply_heatmap(original_pil, cam):
    img_array = np.array(original_pil.resize((224, 224)))
    cam_resized = cv2.resize(cam, (224, 224))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (0.6 * img_array + 0.4 * heatmap).astype(np.uint8)

    pil_overlay = Image.fromarray(overlay)
    buffer = io.BytesIO()
    pil_overlay.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return b64

# ─────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────

@app.get("/")
def home():
    return {
        "message" : "Deepfake Detector API is running! ✅",
        "docs"    : "Visit /docs to test the API"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents)).convert("RGB")

        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs     = model(input_tensor)
            probs       = torch.softmax(outputs, dim=1)[0]
            pred_idx    = probs.argmax().item()
            confidence  = probs[pred_idx].item() * 100

        label = CLASSES[pred_idx]

        input_tensor.requires_grad = True
        cam      = gradcam.generate(input_tensor, pred_idx)
        heatmap  = apply_heatmap(image, cam)

        return JSONResponse({
            "prediction"  : label,
            "confidence"  : round(confidence, 2),
            "is_fake"     : label == "fake",
            "heatmap"     : heatmap,
            "message"     : f"This image is {label.upper()} with {confidence:.1f}% confidence"
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)