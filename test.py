import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    import torchvision.models as models
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = models.efficientnet_b0(pretrained=True)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(1280, 2)
    )
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model.to(DEVICE)

def predict_image(model, image_path):
    transforms_predict = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    image = Image.open(image_path).convert('RGB')
    tensor = transforms_predict(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        covid_prob = float(probabilities[0][0]) * 100
        normal_prob = float(probabilities[0][1]) * 100
        print(f"Prediction: {'COVID' if prediction.item() == 0 else 'Normal'}")
        print(f"Confidence: {confidence.item()*100:.2f}%")
        print(f"COVID Probability: {covid_prob:.2f}%")
        print(f"Normal Probability: {normal_prob:.2f}%")

if __name__ == "__main__":
    model_path = "best_covid_model.pth"
    image_path = "dataset/test/covid/COVID-708.png"
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        sys.exit(1)
    model = load_model(model_path)
    predict_image(model, image_path)