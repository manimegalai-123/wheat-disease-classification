import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

# Load model
model = EfficientNet.from_name('efficientnet-b0', num_classes=3)
model.load_state_dict(torch.load('wheat_leaf_model_efficientnet.pth', map_location=torch.device('cpu')))
model.eval()

# Class names
CLASS_NAMES = ['Healthy', 'Septoria', 'Leaf Rust']

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# UI
st.title("🌿 Wheat Leaf Disease Detection")
uploaded_file = st.file_uploader("Upload a leaf image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.nn.functional.softmax(output, dim=1)[0][prediction].item() * 100

    st.success(f"Predicted: **{CLASS_NAMES[prediction]}** ({confidence:.2f}%)")
