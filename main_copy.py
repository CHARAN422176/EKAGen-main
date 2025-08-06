import torch
from models import caption
from ptflops import get_model_complexity_info

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, config, device):
        super().__init__()
        self.model = model
        self.config = config
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = self.config.max_position_embeddings
        target = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        target_mask = torch.ones_like(target).to(self.device)
        class_feature = torch.randn(batch_size, self.config.num_classes).to(self.device)
        if class_feature.dim() == 1:
            class_feature = class_feature.unsqueeze(0)
        return self.model(x, target, target_mask, class_feature)

def compute_flops_params(config):
    device = torch.device(config.device)
    model, _ = caption.build_model(config)
    model.to(device)
    model.eval()
    # Wrap model for structure compatibility
    wrapped_model = ModelWrapper(model, config, device).to(device)
    input_res = (3, config.image_size, config.image_size)
    with torch.cuda.device(0 if 'cuda' in config.device else -1):
        macs, params = get_model_complexity_info(
            wrapped_model, input_res, as_strings=True,
            print_per_layer_stat=False, verbose=False
        )
    print(f"\n✅ FLOPs and Parameters for EKAGen:")
    print(f"{'Input Resolution:':<30} {input_res}")
    print(f"{'Computational Complexity:':<30} {macs}")
    print(f"{'Number of Parameters:':<30} {params}\n")

# Example usage in your main()
# Only run this if you want FLOPs/params and not training:
if __name__ == "__main__":
    # ... argparse block as before ...
    config = parser.parse_args()
    if config.mode == "flops":
        compute_flops_params(config)
        exit(0)
