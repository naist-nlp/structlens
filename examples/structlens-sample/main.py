import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from structlens import StructLens


def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = "Language exhibits inherent structures, a property that explains both language acquisition and language change."
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )
    outputs = model(
        **inputs,
        return_dict_in_generate=True,
        output_hidden_states=True,
        return_token_timestamps=True,
    )

    residuals: tuple = (
        outputs.hidden_states
    )  # Tuple of length 1 + num_layers (input embedding + num_layers)
    # residuals[i]: Tensor of shape (batch_size, num_tokens, hidden_size)

    residuals_tensor = torch.stack(
        residuals
    )  # (num_layers, batch_size, num_tokens, hidden_size)
    # Ignore the batch dimension of each layer
    representations = residuals_tensor.squeeze(
        1
    )  # (num_layers, num_tokens, hidden_size)

    struct_lens = StructLens()
    st_list = struct_lens(representations)
    for i, st in enumerate(st_list):
        print("layer: ", i)
        print("argmax_heads: ", st["argmax_heads"])


if __name__ == "__main__":
    main()
