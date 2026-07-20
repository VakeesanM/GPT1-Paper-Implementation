from components.transformer import Transformer
from transformers import AutoTokenizer
import torch
from pathlib import Path


def load_model():
    """Loads Model with pretrained weights"""
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    model = Transformer().to(device)

    project_root = Path(__file__).resolve().parent.parent
    weight_path = project_root / "training" / "model.pt"


    weights = torch.load(weight_path, weights_only=True, map_location=torch.device(device))
    model.load_state_dict(weights)
    model.eval()
    return model

def load_tokenizer():
    """Loads the GPT2 Tokenizer from huggingface"""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return tokenizer