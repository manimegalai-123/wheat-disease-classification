import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

# Define number of classes
CLASS_NAMES = ['Healthy', 'Septoria', 'Leaf Rust']

# Load EfficientNet and manually replace the classifier (_fc)
model = EfficientNet.from_name('efficientnet-b0')
num_classes = 3
model._fc = torch.nn.Linear(in_features=model._fc.in_features, out_features=num_classes)
# Load the trained model weights
model.load_state_dict(torch.load("wheat_leaf_model_efficientnet.pth", map_location=torch.device('cpu')))
model.eval()

# Image path
img_path = "test1.png"  # Replace with your image path

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load and prepare image
image_pil = Image.open(img_path).convert("RGB")
image_tensor = transform(image_pil).unsqueeze(0)  # Add batch dimension

# Predict
with torch.no_grad():
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_class = CLASS_NAMES[predicted.item()]
    confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item() * 100

# Show image and prediction
plt.imshow(image_pil)
plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)")
plt.axis('off')
plt.show()

# Print prediction
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
