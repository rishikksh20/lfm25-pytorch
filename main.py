from pathlib import Path
from config import LFM2Config
from huggingface_hub import snapshot_download
from lfm2 import LFM2ForCausalLM
from safetensors.torch import load_file
from sampling import advance_decoding
from tokenizer import LFM2Tokenizer
import torch
import os
from utils import load_weights_into_lfm2

# Configuration

device = torch.device("cpu")
repo_id = "LiquidAI/LFM2.5-2.6B"
local_dir = Path(repo_id).parts[-1]
config = LFM2Config()
# Instantiate the combined model
model = LFM2ForCausalLM(config)

print(f"Downloading model snapshot from {repo_id}...")
try:
    # 1. Download weights
    model_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=["*.safetensors", "*.json"]
    )

    # 2. Load all shards into memory
    weights_dict = {}
    safetensor_files = sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")])

    print(f"Found {len(safetensor_files)} weight shards. Loading...")
    for f in safetensor_files:
        shard = load_file(os.path.join(model_path, f))
        weights_dict.update(shard)

    # 3. Fix Vocabulary Mismatch
    hf_vocab_size = weights_dict["model.embed_tokens.weight"].shape[0]
    if model.vocab_size != hf_vocab_size:
        print(f"Adjusting model vocab_size from {model.vocab_size} to {hf_vocab_size} to match weights.")
        model.vocab_size = hf_vocab_size
        model.config.vocab_size = hf_vocab_size
        model.embed_tokens = torch.nn.Embedding(
            hf_vocab_size,
            model.config.hidden_size,
            padding_idx=model.config.pad_token_id,
            dtype=model.config.dtype,
        )
        with torch.no_grad():
            model.embed_tokens.weight[model.config.pad_token_id].zero_()
        # If weights are tied, we must update the lm_head reference too
        if model.config.tie_word_embeddings:
            model.tie_weights()
        else:
            model.lm_head = torch.nn.Linear(
                model.config.hidden_size,
                hf_vocab_size,
                bias=False,
                dtype=model.config.dtype,
            )

    print("Mapping weights into the custom LFM2 model...")
    load_weights_into_lfm2(model, weights_dict)

    # Move model to device and prepare for inference
    model.to(device=device, dtype=config.dtype)
    model.eval()
    print("Model is ready for inference.")

except Exception as e:
    raise RuntimeError(f"An error occurred during loading: {e}") from e


# --- Usage Code ---
tokenizer_file_path = os.path.join(local_dir, "tokenizer.json")
if not os.path.exists(tokenizer_file_path):
    from huggingface_hub import hf_hub_download
    tokenizer_file_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", local_dir=local_dir)

# Initialize Tokenizer
tokenizer = LFM2Tokenizer(
    tokenizer_file_path=tokenizer_file_path,
    tokenizer_config_path=os.path.join(local_dir, "tokenizer_config.json"),
    model_config=config,
)

# Prepare Prompt
prompt = "Please explain the climate change and how it impacts our future."
formatted_prompt = tokenizer.apply_chat_template(prompt)

# Encode
input_token_ids = tokenizer.encode(formatted_prompt)
input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

print(f"Raw Prompt: {prompt}")
print(f"Formatted Prompt:\n{formatted_prompt}")
print(f"Token IDs count: {len(input_token_ids)}")
print(f"Input Tensor Shape: {input_token_ids_tensor.shape}")

# Execution loop with streaming output
print("Assistant: ", end="", flush=True)
for token in advance_decoding(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=200,
    eos_token_id=tokenizer.eos_id,
    temperature=0.1,
    top_k=50,
    top_p=None,
    repetition_penalty=1.1,
    window_size=64
):
    token_id = token.squeeze().tolist()
    # Handle case where squeeze might return a scalar
    token_list = [token_id] if isinstance(token_id, int) else token_id
    print(tokenizer.decode(token_list), end="", flush=True)
