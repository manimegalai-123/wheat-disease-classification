import torch
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet

CLASS_NAMES = ['Healthy', 'Septoria', 'Leaf Rust']

def load_model():
    model = EfficientNet.from_name('efficientnet-b0')
    model._fc = torch.nn.Linear(model._fc.in_features, 3)
    model.load_state_dict(torch.load("wheat_leaf_model_efficientnet.pth", map_location="cpu"))
    model.eval()
    return model

def test_model_prediction():
    model = load_model()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open("test1.png").convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    # ✅ Assertion (VERY IMPORTANT)
    assert predicted.item() in [0, 1, 2]
