import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# Загрузка модели
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Замените на количество классов в вашей задаче
    model.load_state_dict(torch.load("enhanced_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Преобразования для изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Главная функция приложения
def main():
    st.title("Распознавание дорожных ям")
    st.write("Загрузите изображение дороги, чтобы модель могла предсказать наличие ямы.")

    # Загрузка изображения
    uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Загруженное изображение.', use_column_width=True)
        st.write("")
        st.write("Обрабатываем изображение...")

        # Преобразование изображения
        img_tensor = transform(image).unsqueeze(0)

        # Загрузка модели и предсказание
        model = load_model()
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            label = "Яма" if predicted.item() == 1 else "Нет ямы"

        st.write(f"Предсказание: {label}")

if __name__ == "__main__":
    main()
фзз
