# LFM2.5 2.6B: Liquid Foundation Models

A minimal PyTorch implementation of
[`LiquidAI/LFM2.5-2.6B`](https://huggingface.co/LiquidAI/LFM2.5-2.6B).
LFM2.5 uses the LFM2 hybrid architecture and updates the vocabulary, context
length, RoPE base, tokenizer, and post-training behavior.

📖 LMF2 Docs: [lfm2 technical docs](https://arxiv.org/pdf/2511.23404)

## Setup

Python 3.11 and all runtime dependencies are managed with `uv`.

```shell
uv sync
uv run python main.py
```

The first run downloads the two Hugging Face weight shards. `main.py` uses the
CPU by default; change `device` to `torch.device("cuda")` when CUDA is available.

## LFM2.5-2.6B Config

The values below come from the checkpoint's
[`config.json`](https://huggingface.co/LiquidAI/LFM2.5-2.6B/blob/main/config.json).

```python
import torch

LFM25_CONFIG = {
    "vocab_size": 128_000,
    "context_length": 131_072,
    "emb_dim": 2_048,
    "n_heads": 32,
    "n_kv_heads": 8,
    "n_layers": 30,
    "hidden_dim": 10_752,
    "head_dim": 64,
    "conv_kernel_size": 3,
    "norm_eps": 1e-5,
    "rope_base": 10_000_000.0,
    "bos_token_id": 124_894,
    "eos_token_id": 124_900,
    "pad_token_id": 124_893,
    "dtype": torch.bfloat16,
}
```

The model contains 30 hybrid decoder blocks:

- 22 causal, double-gated, depthwise convolution blocks with kernel size 3.
- 8 grouped-query attention blocks with 32 query heads and 8 key/value heads.
- Every block contains pre-normalization, a residual mixer, and a SwiGLU
  feed-forward network.
- Attention uses query/key RMSNorm, 64-dimensional heads, causal masking, and
  RoPE with a base of 10,000,000.
- Token embeddings and the language-model output head share weights.
- Autoregressive generation caches attention keys/values and convolution state.

The full layer schedule is defined in [`config.py`](config.py). Architecture
references are available in the
[`LFM2 technical report`](https://arxiv.org/abs/2511.23404).

## Code Pointers

| File | Responsibility |
| --- | --- |
| [`config.py`](config.py) | LFM2.5-2.6B dimensions, layer schedule, token IDs, RoPE, and dtype |
| [`lfm2.py`](lfm2.py) | Decoder blocks, model assembly, causal masks, and hybrid cache handling |
| [`modules.py`](modules.py) | RMSNorm, RoPE, SwiGLU, and causal double-gated convolution |
| [`attention.py`](attention.py) | Grouped-query causal attention and attention KV cache |
| [`utils.py`](utils.py) | Hugging Face checkpoint-to-model weight mapping |
| [`tokenizer.py`](tokenizer.py) | Special tokens and the LFM2.5 reasoning chat template |
| [`sampling.py`](sampling.py) | Temperature, top-k, top-p, repetition penalty, and streaming decode |
| [`main.py`](main.py) | End-to-end checkpoint download, loading, tokenization, and inference |

## Load The Model

Instantiate the local architecture using `LFM2Config`. All model parameters use
the `torch.bfloat16` dtype declared by `config.dtype`.

```python
import torch

from config import LFM2Config
from lfm2 import LFM2ForCausalLM

config = LFM2Config()
model = LFM2ForCausalLM(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Load Hugging Face Weights

The checkpoint is sharded, so download the snapshot and merge its Safetensors
files before mapping them into the local model.

```python
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from utils import load_weights_into_lfm2

repo_id = "LiquidAI/LFM2.5-2.6B"
local_dir = Path(repo_id).name

model_path = Path(
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=["*.safetensors", "*.json"],
    )
)

weights_dict = {}
for weights_file in sorted(model_path.glob("*.safetensors")):
    weights_dict.update(load_file(str(weights_file)))

load_weights_into_lfm2(model, weights_dict)
model.to(device=device, dtype=config.dtype)
model.eval()
del weights_dict
```

Loading all shards into one dictionary temporarily requires memory for both the
checkpoint tensors and the instantiated model.

## Load The Tokenizer

`snapshot_download` also downloads `tokenizer.json` and
`tokenizer_config.json`. The wrapper validates that its BOS, EOS, and padding
IDs match the model config.

```python
from tokenizer import LFM2Tokenizer

tokenizer = LFM2Tokenizer(
    tokenizer_file_path=str(model_path / "tokenizer.json"),
    tokenizer_config_path=str(model_path / "tokenizer_config.json"),
    model_config=config,
)
```

LFM2.5-2.6B is a reasoning model. Its chat template starts every assistant turn
with `<think>`:

```text
<|startoftext|><|im_start|>user
What is climate change?<|im_end|>
<|im_start|>assistant
<think>
```

See the official
[`chat_template.jinja`](https://huggingface.co/LiquidAI/LFM2.5-2.6B/blob/main/chat_template.jinja)
for multi-turn, tool-calling, and structured-content formatting.

## Inference And Sampling

The recommended checkpoint defaults are temperature 0.1, top-k 50, and
repetition penalty 1.1.

```python
from sampling import advance_decoding

prompt = "Please explain climate change and how it impacts our future."
formatted_prompt = tokenizer.apply_chat_template(prompt)
input_token_ids = tokenizer.encode(formatted_prompt)
input_ids = torch.tensor(input_token_ids, device=device).unsqueeze(0)

print(f"Prompt: {prompt}")
print("Assistant: ", end="", flush=True)

for token in advance_decoding(
    model=model,
    token_ids=input_ids,
    max_new_tokens=512,
    eos_token_id=tokenizer.eos_id,
    temperature=0.1,
    top_k=50,
    top_p=None,
    repetition_penalty=1.1,
    window_size=64,
):
    token_id = token.squeeze().tolist()
    token_ids = [token_id] if isinstance(token_id, int) else token_id
    print(tokenizer.decode(token_ids), end="", flush=True)
```

The complete runnable version is in [`main.py`](main.py).
