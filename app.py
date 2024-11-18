import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Загрузка сохранённой модели
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Количество классов = 2
    model.load_state_dict(torch.load("enhanced_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Загрузка модели
model = load_model()

# Интерфейс приложения
st.title("Pothole Detection App")
st.write("Upload an image to classify it as `Pothole` or `Normal`.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Отображение загруженного изображения
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Подготовка изображения
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(image).unsqueeze(0)  # Добавляем batch dimension

    # Предсказание
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_names = ["Normal", "Pothole"]
        st.write(f"Prediction: **{class_names[predicted[0]]}**")
