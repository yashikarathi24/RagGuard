
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Base pretrained model
MODEL_NAME = "answerdotai/ModernBERT-base"

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path):
    """
    Load the trained ModernBERT model and tokenizer.

    Supports:
    1. A Hugging Face model directory.
    2. A PyTorch .pt checkpoint containing model weights.
    """

    model_path = Path(model_path)

    # Load tokenizer
    if model_path.is_dir() and (model_path / "tokenizer_config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load complete Hugging Face model directory
    if model_path.is_dir() and (model_path / "config.json").exists():
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path)
        )

    else:
        # Initialize the same base architecture
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            id2label={
                0: "Supported",
                1: "Hallucinated"
            },
            label2id={
                "Supported": 0,
                "Hallucinated": 1
            }
        )

        # Load trained weights
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True
        )

        # Support checkpoints saved as a state_dict
        # or inside a dictionary
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        model.load_state_dict(checkpoint, strict=True)

    # Move model to CPU/GPU
    model.to(DEVICE)

    # Switch to evaluation mode
    model.eval()

    return model, tokenizer
