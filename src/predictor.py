
import torch

from src.preprocessing import prepare_input
from src.model_loader import DEVICE


def predict_hallucination(
    source_text,
    response_text,
    model,
    tokenizer,
    max_length=4096,
    threshold=0.35
):
    """
    Predict whether a generated response is hallucinated
    or supported by the provided source context.
    """

    inputs = prepare_input(
        source_text,
        response_text,
        tokenizer,
        max_length=max_length
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    model.eval()

    with torch.inference_mode():
        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        hallucination_probability = probabilities[0, 1].item()

        predicted_class = (
            1 if hallucination_probability >= threshold else 0
        )

        confidence = probabilities[0, predicted_class].item()

    label = (
        "Hallucinated"
        if predicted_class == 1
        else "Supported"
    )

    return {
        "prediction": label,
        "confidence": confidence,
        "threshold": threshold,
        "probabilities": {
            "supported": probabilities[0, 0].item(),
            "hallucinated": probabilities[0, 1].item()
        }
    }