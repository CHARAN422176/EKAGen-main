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


def build_diagnosisbot(num_classes, detector_weight_path):
    model = create_model(num_classes=num_classes)
    assert os.path.exists(detector_weight_path), "file: '{}' does not exist.".format(detector_weight_path)
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


def main(config):
    print(config)
    device = torch.device(config.device)
    print(f'Initializing Device: {device}')

    if args.mode == "flops":
        from fvcore.nn import FlopCountAnalysis, parameter_count_table

        # Dummy input (3x300x300 as per EkaGen config)
        dummy_input = torch.randn(1, 3, args.image_size, args.image_size).to(device)

        model.eval()
        try:
            print("Analyzing FLOPs and Parameters using fvcore...")
            print(f"Model: {args.exp_name if hasattr(args, 'exp_name') else 'EkaGen'}")
            print(f"Input size: {dummy_input.shape}")
            
            flops = FlopCountAnalysis(model, dummy_input)
            print(flops)
            print("Total FLOPs: {:.2f} GFLOPs".format(flops.total() / 1e9))

            param_table = parameter_count_table(model)
            print(param_table)
        except Exception as e:
            print("Error computing FLOPs:", e)


        return  # Exit after computing FLOPs and params

    # ------------------- Regular Training/Test -------------------
    if os.path.exists(config.thresholds_path):
        with open(config.thresholds_path, "rb") as f:
            thresholds = pickle.load(f)

    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    detector = build_diagnosisbot(config.num_classes, config.detector_weight_path)
    detector.to(device)

    model, criterion = caption.build_model(config)
    criterionKD = SoftTarget(4.0)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    from ptflops import get_model_complexity_info

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (3, args.image_size, args.image_size), as_strings=True,
                                                print_per_layer_stat=False, verbose=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


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

    # diagnosisbot
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

    # mode
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--test_path', type=str, default="")

    config = parser.parse_args()
    main(config)
