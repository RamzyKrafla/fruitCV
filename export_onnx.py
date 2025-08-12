import json
import torch
from torchvision import models

CKPT_PATH = "model/best_resnet18_finetune.pt"  # fallback to best_resnet18.pt if needed
OUT_ONNX = "model/best_resnet18_finetune.onnx"
OUT_CLASSES = "model/classes.json"
IMAGE_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Try to load fine-tuned checkpoint; fallback to the frozen-head one
ckpt = None
try:
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
except FileNotFoundError:
    CKPT_PATH = "model/best_resnet18.pt"
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

classes = ckpt["classes"]
print(f"Loaded checkpoint {CKPT_PATH} with classes: {classes}")

# Build model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE)
model.eval()

# Dummy input
dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

torch.onnx.export(
    model,
    dummy,
    OUT_ONNX,
    input_names=["input"],
    output_names=["logits"],
    opset_version=12,
    do_constant_folding=True,
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
)
print(f"Wrote ONNX model to {OUT_ONNX}")

# Save classes
with open(OUT_CLASSES, "w") as f:
    json.dump({"classes": classes}, f)
print(f"Wrote classes to {OUT_CLASSES}")
