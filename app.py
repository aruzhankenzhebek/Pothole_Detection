import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn

# Определение класса EnhancedModel
class EnhancedModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super(EnhancedModel, self).__init__()
        # Используем все слои, кроме последних двух
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        last_conv_out_channels = 512  # Для ResNet18 это фиксированное значение
        self.bn = nn.BatchNorm2d(last_conv_out_channels)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(last_conv_out_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.bn(x)  # Batch Normalization
        x = self.fc(x)
        return x

# Загрузка сохранённой модели
@st.cache(allow_output_mutation=True)
def load_model():
    num_classes = 2
    base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = EnhancedModel(base_model, num_classes)
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
        st.write(f"Prediction: **{class_names[predicted.item()]}**")
