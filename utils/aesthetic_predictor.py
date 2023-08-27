import torch
import torch.nn as nn
from PIL import Image
from urllib.request import urlretrieve
import open_clip
import os

model_path = os.environ.get("MODELS_PATH", "..")


def get_aesthetic_model(clip_model="vit_l_14"):
    """Load the aesthetic model."""
    path_to_model = os.path.join(model_path, f"sa_0_4_{clip_model}_linear.pth")
    if not os.path.exists(path_to_model):
        os.makedirs(model_path, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"
            + clip_model
            + "_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)

    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError("Invalid clip_model value")

    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m

amodel = get_aesthetic_model(clip_model="vit_l_14")
amodel.eval()
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai"
)

def score_image(image_path):
    """ Score an image. """
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        prediction = amodel(image_features)
    return prediction