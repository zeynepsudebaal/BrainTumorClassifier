import torch
import gradio as gr
from torchvision import transforms
from PIL import Image

# TorchScript modeli yükle
model = torch.jit.load("brain_tumor_model-ZSB.pt", map_location="cpu")
model.eval()

# FastAI eğitimde kullandığın normalizasyon değerleri (imagenet_stats)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Preprocessing pipeline (notebooktakiyle uyumlu)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

# Sınıf isimleri elle yaz veya dosyadan oku
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']  # notebooktan al

def predict(img: Image.Image):
    # Görüntüyü dönüştür
    x = transform(img).unsqueeze(0)  # batch dimension ekle
    with torch.no_grad():
        outputs = model(x)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        probs = probs.cpu().numpy()[0]
    # Sonuçları dict formatında döndür
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=len(labels)),
    title="Brain Tumor Classifier",
    description="Upload a brain MRI scan to detect tumor type.",
    theme="default"
)

interface.launch()
