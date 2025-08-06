import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
import argparse
import os
from models import utils, caption
from datasets import xray
from utils.engine import train_one_epoch, evaluate
from models.model import swin_tiny_patch4_window7_224 as create_model
from utils.stloss import SoftTarget

# Import ptflops here (make sure ptflops is installed)
try:
    from ptflops import get_model_complexity_info
except ImportError:
    print("Error: ptflops is not installed. Run `pip install ptflops` to enable FLOPs calculation.")
    exit(1)


def build_diagnosisbot(num_classes, detector_weight_path):
    model = create_model(num_classes=num_classes)
    assert os.path.exists(detector_weight_path), "file: '{}' dose not exist.".format(detector_weight_path)
    model.load_state_dict(torch.load(detector_weight_path, map_location=torch.device('cpu')), strict=True)
    for k, v in model.named_parameters():
        v.requires_grad = False
    return model


def build_tmodel(config, device):
    tmodel, _ = caption.build_model(config)
    print("Loading teacher model Checkpoint...")
    tcheckpoint = torch.load(config.t_model_weight_path, map_location='cpu')
    tmodel.load_state_dict(tcheckpoint['model'])
    tmodel.to(device)
    return tmodel


# ModelWrapper to provide dummy inputs for FLOPs calculation
class ModelWrapper(torch.nn.Module):
    def __init__(self, model, config, device):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.config = config
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = self.config.max_position_embeddings

        target = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        target_mask = torch.ones_like(target).to(self.device)
        class_feature = torch.randn(batch_size, self.config.num_classes).to(self.device)

        # Ensure class_feature is at least 2D to avoid iteration issues
        if class_feature.dim() == 1:
            class_feature = class_feature.unsqueeze(0)

        return self.model(x, target, target_mask, class_feature)


def main(config):
    print(config)
    device = torch.device(config.device)
    print(f'Initializing Device: {device}')

    # Build model
    model, criterion = caption.build_model(config)
    model.to(device)
    model.eval()

    # FLOPs and Params mode
    if config.mode == "flops":
        input_res = (3, config.image_size, config.image_size)
        wrapped_model = ModelWrapper(model, config, device).to(device)
        with torch.cuda.device(0 if 'cuda' in config.device else -1):
            macs, params = get_model_complexity_info(
                wrapped_model, input_res, as_strings=True,
                print_per_layer_stat=False, verbose=False
            )
        print(f"\n✅ FLOPs and Parameters for model '{model.__class__.__name__}':")
        print(f"{'Input Resolution:':<30} {input_res}")
        print(f"{'Computational Complexity:':<30} {macs}")
        print(f"{'Number of Parameters:':<30} {params}\n")
        return  # Exit after flops mode

    # --- The rest of your original main() content for training/testing ---
    # If you want, I can help incorporate the rest; just let me know.
    print("Only FLOPs mode is implemented in this copy.")  # Placeholder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr_drop', type=int, default=20)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Backbone
    parser.add_argument('--backbone', type=str, default='resnet101')
    parser.add_argument('--position_embedding', type=str, default='sine')
    parser.add_argument('--dilation', type=bool, default=True)
    # Basic
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--clip_max_norm', type=float, default=0.1)

    # Transformer
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--pad_token_id', type=int, default=0)
    parser.add_argument('--max_position_embeddings', type=int, default=128)
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--vocab_size', type=int, default=4253)
    parser.add_argument('--start_token', type=int, default=1)
    parser.add_argument('--end_token', type=int, default=2)

    parser.add_argument('--enc_layers', type=int, default=6)
    parser.add_argument('--dec_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--pre_norm', type=int, default=True)

    # DiagnosisBot
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('--thresholds_path', type=str, default="./datasets/thresholds.pkl")
    parser.add_argument('--detector_weight_path', type=str, default="./weight_path/diagnosisbot.pth")
    parser.add_argument('--t_model_weight_path', type=str, default="./weight_path/mimic_t_model.pth")
    parser.add_argument('--knowledge_prompt_path', type=str, default="./knowledge_path/knowledge_prompt_mimic.pkl")

    # ADA
    parser.add_argument('--theta', type=float, default=0.4)
    parser.add_argument('--gamma', type=float, default=0.4)
    parser.add_argument('--beta', type=float, default=1.0)

    # Delta
    parser.add_argument('--delta', type=float, default=0.01)

    # Dataset
    parser.add_argument('--image_size', type=int, default=300)
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr')
    parser.add_argument('--anno_path', type=str, default='../dataset/mimic_cxr/annotation.json')
    parser.add_argument('--data_dir', type=str, default='../dataset/mimic_cxr/images300')
    parser.add_argument('--limit', type=int, default=-1)

    # Mode (train, test, or flops)
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--test_path', type=str, default="")

    config = parser.parse_args()
    main(config)
