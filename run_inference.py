import os
import torch
from PIL import Image
from torchvision import transforms
import sys
sys.path.append('/kaggle/working')
# from datasets.vocab import create_vocab



from models.caption import build_model  # your captioning model
# from datasets.vocab import create_vocab  # ensure it gives `itos` mapping
def create_vocab(anno_path):
    # Minimal dummy vocab object for inference
    class Vocab:
        def __init__(self):
            self.itos = {
                0: "<pad>",
                1: "<start>",
                2: "<end>",
                3: "a",
                4: "normal",
                5: "chest"
            }

        def __getitem__(self, index):
            return self.itos.get(index, "<unk>")

    return Vocab()

from modules.utils import nested_tensor_from_tensor_list

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. Set image folder and model path ---
image_folder = "/kaggle/input/CXR3552_IM-1741/"  # Your input image folder
model_path = "/kaggle/working/weight_path/diagnosisbot.pth"

# --- 2. Preprocess images ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_images(folder):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'):
            img = Image.open(os.path.join(folder, f)).convert('RGB')
            imgs.append(transform(img))
    return imgs

# --- 3. Load model ---
from config import Config  # your configuration class
model, _ = build_model(Config())
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model.eval()

# --- 4. Vocabulary ---
vocab = create_vocab()  # returns {'itos': list, 'stoi': dict}
itos = vocab['itos']
eos_idx = vocab['stoi']['<eos>']

# --- 5. Inference function ---
@torch.no_grad()
def generate_report(images):
    samples = nested_tensor_from_tensor_list(images).to(device)
    class_feature = torch.zeros(1, Config.hidden_dim).to(device)  # dummy class_feature if not used
    target = torch.zeros(1, 1, dtype=torch.long).to(device)  # start token is 0
    target_mask = torch.ones(1, 1).bool().to(device)
    
    report = []
    for _ in range(100):
        out = model(samples, target.permute(1, 0), target_mask.permute(1, 0), class_feature)
        prob = out[:, -1, :]  # last token
        next_word = prob.argmax(-1).unsqueeze(0)
        word_idx = next_word.item()
        if word_idx == eos_idx:
            break
        report.append(itos[word_idx])
        target = torch.cat([target, next_word.unsqueeze(0)], dim=1)
        target_mask = torch.ones_like(target).bool()
    
    return ' '.join(report)

# --- 6. Run ---
images = load_images(image_folder)
if len(images) < 2:
    raise ValueError("Need at least 2 X-ray views for this model")

report = generate_report(images)
print("Generated Report:\n", report)
