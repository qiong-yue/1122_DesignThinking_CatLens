import base64
import pickle
import PIL.Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torch
import torchvision
from typing import Dict, List
import torch.utils.data.dataset
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
import torch.nn as nn
import gradio as gr
from playwright.sync_api import sync_playwright
import pandas as pd

# metadata
k_fold = 5

# data
data_tranforms = transforms.Compose(
    [
            transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            # transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

dataDir = "Cat Breed Dataset"

full_dataset = torchvision.datasets.ImageFolder(dataDir)

class_names = full_dataset.classes

del full_dataset

# load model
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def build_model(num_classes):
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(512, num_classes),
    )
    model.to(device)
    return model

model_weights = [torch.load("inceptionv3_tree{}.pth".format(i), device) for i in range(5)]

class ModelsFromModelWeights(torch.nn.Module):
    def __init__(self, model_weights: List[Dict[str, torch.Tensor]]):
        super().__init__()
        self.model_weights = model_weights
        self.models = [build_model(len(class_names)) for _ in range(len(model_weights))]
        for model, weights in zip(self.models, model_weights):
            model.load_state_dict(weights)
            model.eval()

    def forward(self, img: torch.Tensor):
        outputs = [model(img) for model in self.models]  # 獲取所有模型的輸出
        outputs = torch.cat(outputs, dim=1)  # 沿著特徵維度拼接輸出
        return outputs

with open(f'tree_{k_fold}fold.pkl', 'rb') as f:
    tree: RandomForestClassifier = pickle.load(f)

# predict cat breed
def img_to_breed(img: PIL.Image.Image | np.ndarray):
    cnn = ModelsFromModelWeights(model_weights)
    img: torch.Tensor = data_tranforms(img)
    img = torch.unsqueeze(img, 0)
    outputs = cnn(img.to(device))
    preds = tree.predict(outputs.cpu().detach().numpy())
    return class_names[preds[0]]

# introduction to cats
cat_intro_df = pd.read_csv("./cat_intro.csv")

def cat_intro_from_breed_en(breed_en: str) -> tuple[str, bytes]:
    """從breed_en、breed_zh、url三個欄位中，透過breed_en找到對應的資料，並回傳該品種的breed_zh、url欄位所指定的網頁的HTML內容"""
    cat_info = cat_intro_df.loc[cat_intro_df["breed_en"] == breed_en]
    if len(cat_info) == 0:
        return breed_en, b""
    else:
        # html = requests.get(cat_info["url"].values[0]).text
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 800, "height": 600})
                page.goto(cat_info["url"].values[0])

                try:
                    # 等待 cookie banner 出現
                    banner = page.wait_for_selector("#onetrust-banner-sdk", state="attached", timeout=3000)
                    # 設置 banner 的 style 將其隱藏或移除
                    page.evaluate("banner => banner.style.display = 'none'", banner)
                except:
                    pass

                screenshot = page.screenshot(full_page=True)
            return cat_info["breed_zh"].values[0], screenshot
        except:
            return breed_en, b""

# combine for gradio
def predict(img: PIL.Image.Image):
    breed_en = img_to_breed(img)
    breed_zh, screenshot = cat_intro_from_breed_en(breed_en)
    screenshot = base64.b64encode(screenshot).decode('utf-8')
    scrollable_screenshot = f'<div style="overflow-y: auto; height: 600px;"><img src="data:image/png;base64,{screenshot}" alt="Cat Breed Info"></div>'
    return breed_zh, scrollable_screenshot

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Text(label="預測的品種"), gr.HTML(label="品種介紹")],
    title="Cat Lens"
)

iface.launch(share=True)