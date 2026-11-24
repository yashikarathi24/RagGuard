
def prepare_input(source_text, response_text, tokenizer, max_length=4096):
    """
    Prepare source context and generated response
    for the ModernBERT hallucination detector.
    """

    # Validate inputs
    if not source_text or not source_text.strip():
        raise ValueError("Source context cannot be empty.")

    if not response_text or not response_text.strip():
        raise ValueError("Generated response cannot be empty.")

    # Basic whitespace cleaning
    source_text = " ".join(source_text.split())
    response_text = " ".join(response_text.split())

    # Tokenize source and response as a paired input
    encoded_input = tokenizer(
        source_text,
        response_text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Return tensors required by ModernBERT
    return {
        "input_ids": encoded_input["input_ids"],
        "attention_mask": encoded_input["attention_mask"]
    }
