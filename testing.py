import torch
from torchvision import transforms, models
from PIL import Image
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = val_transforms(img).unsqueeze(0).to(DEVICE)

    # Load model
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(num_features, len(CLASS_NAMES))
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        output = model(img_tensor)
        pred_label = output.argmax(1).item()
    return CLASS_NAMES[pred_label]


# --- SIMPLE USAGE ---
# Set your image path here:
image_path = "image2.jpg"  # <-- Change this to your image file

MODEL_PATH = "best_soyabean_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    "Healthy_Soyabean",
    "Soyabean_Mosaic",
    "Soyabean_Rust",
    "Soyabean_Spectoria_Brown_Spot"
]
val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

pred = predict_image(image_path)
print(f"Predicted class: {pred}")
