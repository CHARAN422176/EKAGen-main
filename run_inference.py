import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
import argparse
import os
from PIL import Image
from torchvision import transforms

# We assume the following files are in your project structure
from models import utils, caption
from datasets import xray
from utils.engine import train_one_epoch, evaluate
from models.model import swin_tiny_patch4_window7_224 as create_model
from utils.stloss import SoftTarget


def build_diagnosisbot(num_classes, detector_weight_path):
    """Builds and loads the pretrained disease detector model."""
    model = create_model(num_classes=num_classes)
    assert os.path.exists(
        detector_weight_path), f"file: '{detector_weight_path}' does not exist."
    model.load_state_dict(torch.load(
        detector_weight_path, map_location=torch.device('cpu')), strict=True)
    # Freeze detector weights
    for k, v in model.named_parameters():
        v.requires_grad = False
    return model


def build_tmodel(config, device):
    """Builds and loads the pretrained teacher model for knowledge distillation."""
    tmodel, _ = caption.build_model(config)
    print("Loading teacher model Checkpoint...")
    tcheckpoint = torch.load(config.t_model_weight_path, map_location='cpu')
    tmodel.load_state_dict(tcheckpoint['model'])
    tmodel.to(device)
    return tmodel

# +------------------------------------------------------------------+
# | NEW FUNCTION FOR SINGLE IMAGE REPORT GENERATION                  |
# +------------------------------------------------------------------+
def generate_report(model, detector, tokenizer, image_path, device, config):
    """
    Generates and prints a report for a single input image.

    Args:
        model: The main captioning model.
        detector: The disease diagnosis model.
        tokenizer: The tokenizer for converting IDs to text.
        image_path (str): Path to the input image file.
        device: The torch device to run the model on.
        config: The configuration arguments.
    """
    model.eval()
    detector.eval()

    # Define the image transformations.
    # These should match the transformations used during training.
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load and preprocess the image
    print(f"Loading and preprocessing image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    # Add a batch dimension (B, C, H, W) and move to the target device
    image_tensor = transform(image).unsqueeze(0).to(device)

    # --- Generation Logic ---
    # This part replicates the likely inference process. The exact implementation
    # can vary, so this is based on standard transformer-based captioning models.
    # The `evaluate` function in your `utils/engine.py` would have the precise logic.
    with torch.no_grad():
        # Step 1: Get diagnosis predictions from the detector model
        tag_predict = detector(image_tensor)

        # Step 2: Generate the report using a greedy search decoding.
        # The model's encoder processes the image first.
        memory = model.encode(image_tensor, tag_predict)

        # Start the generation with the special [START] token
        # ys shape: (batch_size, 1)
        ys = torch.ones(1, 1).fill_(config.start_token).long().to(device)

        for _ in range(config.max_position_embeddings - 1):
            # The decoder predicts the next token based on the image memory and the tokens generated so far.
            out = model.decode(ys, memory)
            # Get the probabilities for the last predicted token
            prob = out[:, -1]
            # Find the token with the highest probability (greedy search)
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            # Append the new token to our sequence
            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(ys.data).fill_(next_word)],
                           dim=1)

            # If the model predicts the [END] token, we stop generating.
            if next_word == config.end_token:
                break

    # Squeeze the batch dimension and convert tensor to a list of token IDs
    generated_ids = ys.squeeze().tolist()

    # Use the tokenizer to convert the list of IDs back to a human-readable string
    generated_report = tokenizer.decode(
        generated_ids, skip_special_tokens=True)

    print("\n" + "="*40)
    print("          GENERATED REPORT")
    print("="*40)
    print(generated_report)
    print("="*40 + "\n")


def main(config):
    print(config)
    device = torch.device(config.device)
    print(f'Initializing Device: {device}')

    if os.path.exists(config.thresholds_path):
        with open(config.thresholds_path, "rb") as f:
            thresholds = pickle.load(f)

    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build the detector and the main captioning model
    detector = build_diagnosisbot(config.num_classes, config.detector_weight_path)
    detector.to(device)

    model, criterion = caption.build_model(config)
    model.to(device)

    # --- Mode Selection Logic ---
    if config.mode == "train":
        # (The original training code remains unchanged)
        criterionKD = SoftTarget(4.0)
        n_parameters = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
        print(f"Number of params: {n_parameters}")

        param_dicts = [
            {"params": [p for n, p in model.named_parameters(
            ) if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": config.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, lr=config.lr, weight_decay=config.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.lr_drop)

        dataset_train = xray.build_dataset(config, mode='training', anno_path=config.anno_path, data_dir=config.data_dir,
                                           dataset_name=config.dataset_name, image_size=config.image_size,
                                           theta=config.theta, gamma=config.gamma, beta=config.beta)
        dataset_val = xray.build_dataset(config, mode='validation', anno_path=config.anno_path, data_dir=config.data_dir,
                                         dataset_name=config.dataset_name, image_size=config.image_size,
                                         theta=config.theta, gamma=config.gamma, beta=config.beta)
        dataset_test = xray.build_dataset(config, mode='test', anno_path=config.anno_path, data_dir=config.data_dir,
                                          dataset_name=config.dataset_name, image_size=config.image_size,
                                          theta=config.theta, gamma=config.gamma, beta=config.beta)
        print(f"Train: {len(dataset_train)}")
        print(f"Valid: {len(dataset_val)}")
        print(f"Test: {len(dataset_test)}")

        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, config.batch_size, drop_last=True
        )

        data_loader_train = DataLoader(
            dataset_train, batch_sampler=batch_sampler_train, num_workers=config.num_workers,
            collate_fn=dataset_train.collate_fn)
        data_loader_val = DataLoader(dataset_val, config.batch_size,
                                     sampler=torch.utils.data.SequentialSampler(dataset_val), drop_last=False,
                                     collate_fn=dataset_val.collate_fn)
        data_loader_test = DataLoader(dataset_test, config.batch_size,
                                      sampler=torch.utils.data.SequentialSampler(dataset_test), drop_last=False,
                                      collate_fn=dataset_test.collate_fn)

        tmodel = build_tmodel(config, device)
        print("Start Training..")
        for epoch in range(config.start_epoch, config.epochs):
            print(f"Epoch: {epoch}")
            epoch_loss = train_one_epoch(
                model, tmodel, detector, criterion, criterionKD, data_loader_train, optimizer, device,
                config.clip_max_norm, thresholds=thresholds, tokenizer=dataset_train.tokenizer, config=config)
            lr_scheduler.step()
            print(f"Training Loss: {epoch_loss}")

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }, config.dataset_name + "_weight_epoch" + str(epoch) + "_.pth")

            validate_result = evaluate(model, detector, criterion, data_loader_val, device, config,
                                       thresholds=thresholds, tokenizer=dataset_val.tokenizer)
            print(f"validate_result: {validate_result}")
            test_result = evaluate(model, detector, criterion, data_loader_test, device, config,
                                     thresholds=thresholds, tokenizer=dataset_test.tokenizer)
            print(f"test_result: {test_result}")

    elif config.mode == "test":
        # (The original test mode for evaluating on a full dataset remains unchanged)
        if os.path.exists(config.test_path):
            weights_dict = torch.load(
                config.test_path, map_location='cpu')['model']
            model.load_state_dict(weights_dict, strict=False)

        dataset_test = xray.build_dataset(config, mode='test', anno_path=config.anno_path, data_dir=config.data_dir,
                                          dataset_name=config.dataset_name, image_size=config.image_size,
                                          theta=config.theta, gamma=config.gamma, beta=config.beta)
        data_loader_test = DataLoader(dataset_test, config.batch_size,
                                      sampler=torch.utils.data.SequentialSampler(dataset_test), drop_last=False,
                                      collate_fn=dataset_test.collate_fn)

        print("Start Testing on the entire dataset..")
        test_result = evaluate(model, detector, criterion, data_loader_test, device, config,
                               thresholds=thresholds, tokenizer=dataset_test.tokenizer)
        print(f"test_result: {test_result}")

    # +------------------------------------------------------------------+
    # | NEW LOGIC BLOCK FOR INFERENCE MODE                               |
    # +------------------------------------------------------------------+
    elif config.mode == "inference":
        if not config.image_path or not os.path.exists(config.image_path):
            raise ValueError(
                "Please provide a valid path to an image using --image_path for inference mode.")

        # The tokenizer is part of the dataset object and is needed to decode the output.
        # We'll create a dummy dataset object just to get the tokenizer.
        # This assumes you have the original dataset's annotation file.
        print("Building tokenizer...")
        dataset_val = xray.build_dataset(config, mode='validation', anno_path=config.anno_path, data_dir=config.data_dir,
                                         dataset_name=config.dataset_name, image_size=config.image_size,
                                         theta=config.theta, gamma=config.gamma, beta=config.beta)
        tokenizer = dataset_val.tokenizer

        # Load the final model weights for inference
        if os.path.exists(config.test_path):
            print(f"Loading model weights from {config.test_path}")
            weights_dict = torch.load(
                config.test_path, map_location='cpu')['model']
            model.load_state_dict(weights_dict, strict=False)
        else:
            raise ValueError(
                f"No model weights found at {config.test_path}. Please provide a path using --test_path.")

        print("Starting single image inference...")
        # Call the new function to generate the report for your image
        generate_report(model, detector, tokenizer,
                        config.image_path, device, config)


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
    parser.add_argument('--device', default='cuda:0',
                        help='device id (i.e. 0 or 0,1 or cpu)')
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
    parser.add_argument('--thresholds_path', type=str,
                        default="./datasets/thresholds.pkl")
    parser.add_argument('--detector_weight_path', type=str,
                        default="./weight_path/diagnosisbot.pth")
    parser.add_argument('--t_model_weight_path', type=str,
                        default="./weight_path/mimic_t_model.pth")
    parser.add_argument('--knowledge_prompt_path', type=str,
                        default="./knowledge_path/knowledge_prompt_mimic.pkl")

    # ADA
    parser.add_argument('--theta', type=float, default=0.4)
    parser.add_argument('--gamma', type=float, default=0.4)
    parser.add_argument('--beta', type=float, default=1.0)

    # Delta
    parser.add_argument('--delta', type=float, default=0.01)

    # Dataset
    parser.add_argument('--image_size', type=int, default=300)
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr')
    parser.add_argument('--anno_path', type=str,
                        default='../dataset/mimic_cxr/annotation.json')
    parser.add_argument('--data_dir', type=str,
                        default='../dataset/mimic_cxr/images300')
    parser.add_argument('--limit', type=int, default=-1)

    # --- MODIFIED ARGUMENTS ---
    # Added 'inference' to choices and a new argument for the single image path.
    parser.add_argument('--mode', type=str, default="train",
                        choices=['train', 'test', 'inference'])
    parser.add_argument('--test_path', type=str, default="",
                        help="Path to the model weights for testing or inference.")
    parser.add_argument('--image_path', type=str, default="",
                        help="Path to a single image for inference mode.")

    config = parser.parse_args()
    main(config)
