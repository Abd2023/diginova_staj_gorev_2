import os
import cv2
import torch
import pickle
import numpy as np
from torchvision import transforms

# Ensure ObjectDetector is in the namespace so pickle can find it
from src.object_detector import ObjectDetector

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "detector.pth")
LE_PATH = os.path.join(BASE_DIR, "model", "le.pickle")

# Device (CPU since we'll run in Docker by default)
DEVICE = torch.device("cpu")

# ImageNet normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Global variables to cache model and label encoder
_model = None
_le = None
_transform = None

def load_model():
    """Loads the model and label encoder once into global variables."""
    global _model, _le, _transform
    
    if _model is None:
        print("[INFO] Loading label encoder...")
        with open(LE_PATH, "rb") as f:
            _le = pickle.loads(f.read())
            
        print("[INFO] Loading object detector model...")
        from torchvision.models import resnet50
        baseModel = resnet50(weights=None)
        _model = ObjectDetector(baseModel, len(_le.classes_))
        
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        # If the state_dict was saved with a full model, torch.load returns the model directly.
        # But if it returns an OrderedDict, it's a state_dict. Let's handle it securely.
        if isinstance(state_dict, dict):
            _model.load_state_dict(state_dict)
        else:
            _model = state_dict
            
        _model.eval()
        print("[INFO] Setting up transforms...")
        _transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    return _model, _le, _transform

def run_inference(image_bytes: bytes) -> dict:
    """Runs inference on a raw image byte string."""
    model, le, transform = load_model()
    
    # Convert bytes to numpy array then to opencv image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Invalid image data.")
        
    orig = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.transpose((2, 0, 1))
    
    # Preprocess
    image_tensor = torch.from_numpy(image)
    image_tensor = transform(image_tensor).to(DEVICE)
    image_tensor = image_tensor.unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        (boxPreds, labelPreds) = model(image_tensor)
        
    (startX, startY, endX, endY) = boxPreds[0]
    
    # Softmax for probabilities
    labelPreds = torch.nn.Softmax(dim=-1)(labelPreds)
    
    # Get highest probability
    i = labelPreds.argmax(dim=-1).cpu().item()
    label = le.inverse_transform([i])[0]
    confidence = labelPreds[0][i].item()
    
    return {
        "label": label,
        "confidence": confidence,
        "bounding_box": {
            "startX": float(startX),
            "startY": float(startY),
            "endX": float(endX),
            "endY": float(endY)
        }
    }
