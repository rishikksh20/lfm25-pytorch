import torch
import torch.nn.functional as F

def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0, recent_tokens=None):
    """
    Apply temperature, top-k, top-p, and repetition penalty sampling.
    """
    # Temperature scaling
    if temperature != 1.0:
        logits = logits / temperature

    # Repetition penalty (penalize tokens that appeared recently)
    if repetition_penalty != 1.0 and recent_tokens is not None:
        # Convert to set for faster lookup
        unique_tokens = set(recent_tokens.tolist())
        for token in unique_tokens:
            # If logits are negative, multiplying by penalty makes them more negative
            # If positive, dividing makes them smaller.
            if logits[..., token] > 0:
                logits[..., token] /= repetition_penalty
            else:
                logits[..., token] *= repetition_penalty

    # Top-k filtering
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k)
        mask = logits < values[..., -1, None]
        logits = logits.masked_fill(mask, float("-inf"))

    # Top-p (nucleus) filtering
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # Mask tokens where cumulative prob > top_p
        sorted_mask = cumulative_probs > top_p
        # shift mask so at least 1 token is kept
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        # apply mask
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
        logits = logits.masked_fill(mask, float("-inf"))

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)

    # Sample
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

def advance_decoding(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    temperature=1.0,
    top_k=None,
    top_p=None,
    repetition_penalty=1.0,
    window_size=50,
):
    """
    Streaming text generation with advanced sampling.
    """
    model.eval()
    device = next(model.parameters()).device
    token_ids = token_ids.to(device)
    current_token_ids = token_ids
    past_key_values = None

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass: get logits of the last token
            logits, past_key_values = model(
                current_token_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = logits[:, -1, :].float()

            # Advanced sampling
            next_token = sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                recent_tokens=token_ids[:, -window_size:].flatten()
            )

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            yield next_token

            # Update sequence
            token_ids = torch.cat([token_ids, next_token], dim=1)
            current_token_ids = next_token
