import gc
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from safetensors.torch import load_file

from config import LFM2Config
from lfm2 import LFM2ForCausalLM
from src.sampling import advance_decoding
from src.tokenizer import LFM2Tokenizer
from src.utils import load_weights_into_lfm2


ROOT = Path(__file__).resolve().parent
LOGS_DIR = Path(os.getenv("LFM_LOGS_DIR", str(ROOT / "logs")))
REPO_ID = "LiquidAI/LFM2.5-2.6B"
MODEL_DIR = ROOT / Path(REPO_ID).name
MAX_NEW_TOKENS = int(os.getenv("LFM_MAX_NEW_TOKENS", "1024"))
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model():
    device = pick_device()
    config = LFM2Config()
    model = LFM2ForCausalLM(config)

    model_path = MODEL_DIR
    if not (model_path / "tokenizer.json").exists() or not any(model_path.glob("*.safetensors")):
        model_path = Path(
            snapshot_download(
                repo_id=REPO_ID,
                local_dir=MODEL_DIR,
                allow_patterns=["*.safetensors", "*.json"],
            )
        )

    weights = {}
    for shard_path in sorted(model_path.glob("*.safetensors")):
        weights.update(load_file(shard_path))

    hf_vocab_size = weights["model.embed_tokens.weight"].shape[0]
    if model.vocab_size != hf_vocab_size:
        model.vocab_size = hf_vocab_size
        model.config.vocab_size = hf_vocab_size
        model.embed_tokens = torch.nn.Embedding(
            hf_vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            dtype=config.dtype,
        )
        with torch.no_grad():
            model.embed_tokens.weight[config.pad_token_id].zero_()
        if config.tie_word_embeddings:
            model.tie_weights()
        else:
            model.lm_head = torch.nn.Linear(
                config.hidden_size,
                hf_vocab_size,
                bias=False,
                dtype=config.dtype,
            )

    load_weights_into_lfm2(model, weights)
    model.to(device=device, dtype=config.dtype).eval()
    del weights
    gc.collect()

    tokenizer = LFM2Tokenizer(
        tokenizer_file_path=str(model_path / "tokenizer.json"),
        tokenizer_config_path=str(model_path / "tokenizer_config.json"),
        model_config=config,
    )
    return model, tokenizer, device


print("Loading local LFM2 model...")
model, tokenizer, device = load_model()
print(f"LFM2 is ready on {device}.")

app = FastAPI(title="Local LFM2 Chat")
generation_lock = threading.Lock()
chat_lock = threading.Lock()
active_chats = set()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8_000)


def chat_path(chat_id: UUID) -> Path:
    return LOGS_DIR / f"{chat_id}.json"


def new_chat() -> dict:
    timestamp = utc_now()
    return {
        "version": 1,
        "id": str(uuid4()),
        "model": REPO_ID,
        "created_at": timestamp,
        "updated_at": timestamp,
        "messages": [],
    }


def save_chat(chat: dict) -> None:
    path = chat_path(UUID(chat["id"]))
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(chat, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_chat(chat_id: UUID) -> dict:
    path = chat_path(chat_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Chat not found")
    return json.loads(path.read_text(encoding="utf-8"))


def build_chat_prompt(messages: list[dict]) -> str:
    parts = [tokenizer.bos_token]
    for message in messages:
        if message["role"] == "user":
            parts.append(f"<|im_start|>user\n{message['content']}{tokenizer.eos_token}\n")
        elif message["role"] == "assistant":
            thinking = message.get("thinking", "")
            content = message.get("content", "")
            parts.append(
                f"<|im_start|>assistant\n<think>{thinking}</think>{content}"
                f"{tokenizer.eos_token}\n"
            )
    parts.append("<|im_start|>assistant\n<think>")
    return "".join(parts)


def response_parts(decoded_text: str) -> tuple[str, str]:
    opening_tag = "<think>"
    if opening_tag.startswith(decoded_text):
        decoded_text = ""
    elif decoded_text.startswith(opening_tag):
        decoded_text = decoded_text[len(opening_tag) :]

    if "</think>" not in decoded_text:
        return decoded_text, ""
    return tuple(decoded_text.split("</think>", 1))


def event(event_type: str, **data) -> str:
    return json.dumps({"type": event_type, **data}, ensure_ascii=False) + "\n"


def stream_reply(chat: dict):
    chat_id = UUID(chat["id"])
    prompt = build_chat_prompt(chat["messages"])
    token_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
    generated_ids = []
    emitted_thinking = ""
    emitted_answer = ""
    error_message = None

    try:
        # Generation uses a mutable model cache, so only one reply runs at a time.
        with generation_lock:
            for token in advance_decoding(
                model=model,
                token_ids=token_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=tokenizer.eos_id,
                temperature=0.1,
                top_k=50,
                top_p=None,
                repetition_penalty=1.1,
                window_size=64,
            ):
                token_id = token.squeeze().tolist()
                generated_ids.extend([token_id] if isinstance(token_id, int) else token_id)
                decoded_text = tokenizer.decode(generated_ids)
                thinking, answer = response_parts(decoded_text)

                if thinking.startswith(emitted_thinking):
                    for character in thinking[len(emitted_thinking) :]:
                        yield event("thinking", delta=character)
                elif thinking != emitted_thinking:
                    yield event("thinking_replace", text=thinking)
                emitted_thinking = thinking

                if answer.startswith(emitted_answer):
                    for character in answer[len(emitted_answer) :]:
                        yield event("answer", delta=character)
                elif answer != emitted_answer:
                    yield event("answer_replace", text=answer)
                emitted_answer = answer
    except Exception as error:
        error_message = str(error)
        yield event("error", message="Generation failed.")
    finally:
        assistant_message = {
            "role": "assistant",
            "thinking": emitted_thinking,
            "content": emitted_answer,
            "created_at": utc_now(),
        }
        if error_message:
            assistant_message["error"] = error_message

        with chat_lock:
            latest_chat = load_chat(chat_id)
            latest_chat["messages"].append(assistant_message)
            latest_chat["updated_at"] = utc_now()
            save_chat(latest_chat)
            active_chats.discard(chat_id)

    yield event("done", chat_id=str(chat_id))


@app.get("/api/chats")
def list_chats():
    chats = []
    with chat_lock:
        for path in LOGS_DIR.glob("*.json"):
            try:
                chat_id = UUID(path.stem)
                chat = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, json.JSONDecodeError, OSError):
                continue
            chats.append(
                {
                    "id": str(chat_id),
                    "created_at": chat["created_at"],
                    "updated_at": chat["updated_at"],
                    "message_count": len(chat.get("messages", [])),
                }
            )
    return sorted(chats, key=lambda item: item["updated_at"], reverse=True)


@app.post("/api/chats", status_code=201)
def create_chat():
    chat = new_chat()
    with chat_lock:
        save_chat(chat)
    return chat


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: UUID):
    with chat_lock:
        return load_chat(chat_id)


@app.post("/api/chats/{chat_id}/messages")
def chat(chat_id: UUID, request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be blank")

    with chat_lock:
        if chat_id in active_chats:
            raise HTTPException(status_code=409, detail="This chat is already generating a reply")
        chat_data = load_chat(chat_id)
        timestamp = utc_now()
        chat_data["messages"].append(
            {"role": "user", "content": message, "created_at": timestamp}
        )
        chat_data["updated_at"] = timestamp
        save_chat(chat_data)
        active_chats.add(chat_id)

    return StreamingResponse(
        stream_reply(chat_data),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
