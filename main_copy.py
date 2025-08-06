import torch
import torch.nn as nn
import pickle
from ptflops import get_model_complexity_info

# Dummy backbone - minimal for FLOPs
class DummyBackbone(nn.Module):
    def __init__(self, num_channels=256):
        super().__init__()
        self.num_channels = num_channels
        # Minimal Conv2d network just to match interface
        self.conv = nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((32, 32))
    def forward(self, x):
        # Returns (features_list, pos_embeddings)
        out = self.conv(x)
        out = self.pool(out)
        # Fake output shapes compatible with model
        return [NestedTensor(out, torch.zeros(out.shape[0], out.shape[2], out.shape[3], dtype=torch.bool, device=out.device))], [out]

# Dummy NestedTensor for backbone output
class NestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask
    def decompose(self):
        return self.tensors, self.mask

# Dummy Transformer to simulate forward structure
class DummyTransformer(nn.Module):
    def __init__(self, config, knowledge_prompt):
        super().__init__()
        self.knowledge_prompt = knowledge_prompt
        # These lines minimal; in real, you'd have encoder/decoder for shape
    def forward(self, src, mask, pos_embed, tgt, tgt_mask, class_feature):
        # Simulate class_feature dictionary lookup (no real computation)
        class_feature0 = class_feature[0]  # [num_classes, key_length]
        batch_key_list = [tuple([int(v.item()) for v in item]) for item in class_feature0]
        # Lookup keys: just pick first if real not found
        feature_list = []
        template_key = list(self.knowledge_prompt.keys())[0]
        for key in batch_key_list:
            if key in self.knowledge_prompt:
                feature_list.append(self.knowledge_prompt[key])
            else:
                feature_list.append(self.knowledge_prompt[template_key])
        class_feature_tensor = torch.stack(
            [torch.tensor(v, dtype=torch.float32) if not torch.is_tensor(v) else v.float() for v in feature_list]
        ).to(tgt.device)
        out = tgt[..., :class_feature_tensor.shape[-1]] + class_feature_tensor.unsqueeze(0)
        return out

# Dummy MLP (matches EKAGen structure)
class DummyMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        x = x.mean(dim=-1) if x.ndim > 2 else x
        return self.fc(x)

# Main EKAGen dummy caption model
class DummyCaption(nn.Module):
    def __init__(self, config, knowledge_prompt):
        super().__init__()
        self.backbone = DummyBackbone(num_channels=config.hidden_dim)
        self.input_proj = nn.Conv2d(config.hidden_dim, config.hidden_dim, kernel_size=1)
        self.transformer = DummyTransformer(config, knowledge_prompt)
        self.mlp = DummyMLP(config.hidden_dim, config.vocab_size)
    def forward(self, samples, target, target_mask, class_feature):
        if samples.dim() == 3:
            samples = samples.unsqueeze(0)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        tgt_out = self.transformer(self.input_proj(src), mask, pos[-1], target, target_mask, class_feature)
        out = self.mlp(tgt_out.permute(1, 0, 2) if tgt_out.ndim == 3 else tgt_out)
        return out

# Minimal Config Object for all hyperparams
class Config:
    def __init__(self):
        self.hidden_dim = 256
        self.vocab_size = 760
        self.max_position_embeddings = 128
        self.num_classes = 14
        self.image_size = 300
        self.device = 'cuda:0'
        self.knowledge_prompt_path = '/kaggle/working/knowledge_path/knowledge_prompt_iu.pkl'

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=300)
    parser.add_argument('--vocab_size', type=int, default=760)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--max_position_embeddings', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--knowledge_prompt_path', type=str, required=True)
    args = parser.parse_args()

    # Prepare configuration
    config = Config()
    config.image_size = args.image_size
    config.vocab_size = args.vocab_size
    config.hidden_dim = args.hidden_dim
    config.max_position_embeddings = args.max_position_embeddings
    config.num_classes = args.num_classes
    config.device = args.device
    config.knowledge_prompt_path = args.knowledge_prompt_path

    # Load knowledge prompt dict and deduce key_length
    with open(config.knowledge_prompt_path, 'rb') as f:
        kp = pickle.load(f)
    template_key = list(kp.keys())[0]
    key_length = len(template_key)

    # Build dummy model
    model = DummyCaption(config, kp).to(config.device)
    model.eval()

    # ModelWrapper for ptflops dummy input
    class ModelWrapper(nn.Module):
        def __init__(self, model, config):
            super().__init__()
            self.model = model
            self.config = config
            self.key_length = key_length
            self.num_classes = config.num_classes
            self.device = config.device
            self.template_key = template_key
        def forward(self, x):
            if isinstance(x, (tuple, list)):
                x = x[0]
            if x.dim() == 3:
                x = x.unsqueeze(0)
            batch_size = x.shape[0]
            seq_len = self.config.max_position_embeddings
            target = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), device=self.device)
            target_mask = torch.ones_like(target)
            # All valid keys everywhere
            cf = torch.tensor(
                [list(self.template_key)],
                dtype=torch.long
            ).repeat(batch_size, self.num_classes, 1).to(self.device)
            return self.model(x, target, target_mask, cf)

    wrapped_model = ModelWrapper(model, config).to(config.device)
    input_res = (3, config.image_size, config.image_size)
    with torch.cuda.device(0 if 'cuda' in config.device else -1):
        macs, _ = get_model_complexity_info(
            wrapped_model, input_res, as_strings=True,
            print_per_layer_stat=False, verbose=False
        )
    print(f"\n✅ EKAGen Model FLOPs Only:")
    print(f"{'Input Resolution:':<30} {input_res}")
    print(f"{'Computational Complexity:':<30} {macs}\n")

if __name__ == "__main__":
    main()
