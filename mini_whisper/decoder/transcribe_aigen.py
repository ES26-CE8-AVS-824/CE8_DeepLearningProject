import torch
def _decoder_prefix(tokenizer):
    prompt_tokens = tokenizer.encode("")
    if len(prompt_tokens) < 2:
        raise ValueError("Tokenizer prompt must include at least one prefix token and EOS.")

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = prompt_tokens[-1]

    prefix_tokens = prompt_tokens[:-1]
    return prefix_tokens, eos_token_id


def _decode_step(model, tokens, encoder_output):
    if hasattr(model, "decode_tokens"):
        return model.decode_tokens(tokens, encoder_output)
    return model.decoder(tokens, encoder_output)


def _encode_audio(model, audio):
    if hasattr(model, "encode_audio"):
        return model.encode_audio(audio)
    return model.encoder(audio)


def _resolve_max_new_tokens(model, max_new_tokens, prefix_length):
    decoder_limit = getattr(model, "max_text_len", None)
    if decoder_limit is None:
        return max_new_tokens

    available = decoder_limit - prefix_length
    if available <= 0:
        raise ValueError("Model max_text_len is too small for the decoder prefix.")
    return min(max_new_tokens, available)


def _rank_beam(score, generated_length, length_penalty):
    length = max(generated_length, 1)
    if length_penalty == 0.0:
        return score
    return score / (length ** length_penalty)


def _uses_sampling(temperature, top_k, top_p):
    return temperature != 1.0 or top_k is not None or top_p is not None


def _apply_top_k_filter(logits, top_k):
    if top_k is None:
        return logits
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if top_k >= logits.size(-1):
        return logits

    threshold = torch.topk(logits, k=top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p_filter(logits, top_p):
    if top_p is None:
        return logits
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in the interval (0, 1].")
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    filtered_logits = torch.full_like(logits, float("-inf"))
    filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered_logits


def _sample_next_tokens(logits, temperature=1.0, top_k=None, top_p=None):
    if temperature <= 0.0:
        raise ValueError("temperature must be greater than 0 for sampling.")

    logits = logits / temperature
    logits = _apply_top_k_filter(logits, top_k)
    logits = _apply_top_p_filter(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _finalize_sequences(sequences, eos_token_id, pad_token_id, device):
    trimmed_sequences = []
    max_length = 0

    for sequence in sequences:
        if isinstance(sequence, torch.Tensor):
            token_list = sequence.tolist()
        else:
            token_list = list(sequence)

        if eos_token_id in token_list:
            token_list = token_list[:token_list.index(eos_token_id)]

        trimmed_sequences.append(token_list)
        max_length = max(max_length, len(token_list))

    batch_size = len(trimmed_sequences)
    if max_length == 0:
        return torch.empty((batch_size, 0), dtype=torch.long, device=device)

    output = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long, device=device)
    for index, token_list in enumerate(trimmed_sequences):
        if token_list:
            output[index, :len(token_list)] = torch.tensor(token_list, dtype=torch.long, device=device)
    return output


def transcribe(
    model,
    audio,
    tokenizer,
    DEVICE,
    max_new_tokens=100,
    beam_width=1,
    length_penalty=0.6,
    temperature=1.0,
    top_k=None,
    top_p=None,
):
    """Transcribe a batch of mel spectrograms.

    Beam search is used when ``beam_width > 1``; otherwise decoding is greedy.
    """
    if beam_width > 1:
        if _uses_sampling(temperature, top_k, top_p):
            raise ValueError("Sampling options cannot be combined with beam search in transcribe().")
        return transcribe_beam_search(
            model,
            audio,
            tokenizer,
            DEVICE,
            beam_width=beam_width,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
        )
    if _uses_sampling(temperature, top_k, top_p):
        return transcribe_sampling(
            model,
            audio,
            tokenizer,
            DEVICE,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    audio = audio.to(DEVICE)
    batch_size = audio.shape[0]
    prefix_tokens, eos_token_id = _decoder_prefix(tokenizer)
    prefix_length = len(prefix_tokens)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    max_new_tokens = _resolve_max_new_tokens(model, max_new_tokens, prefix_length)

    decoder_input = torch.tensor(prefix_tokens, dtype=torch.long, device=DEVICE).unsqueeze(0).expand(batch_size, -1)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            encoder_output = _encode_audio(model, audio)
            for _ in range(max_new_tokens):
                logits = _decode_step(model, decoder_input, encoder_output)
                next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
                next_tokens = torch.where(
                    finished,
                    torch.full_like(next_tokens, eos_token_id),
                    next_tokens,
                )

                decoder_input = torch.cat([decoder_input, next_tokens.unsqueeze(1)], dim=1)
                finished |= next_tokens.eq(eos_token_id)
                if finished.all():
                    break
    finally:
        if was_training:
            model.train()

    return _finalize_sequences(decoder_input[:, prefix_length:], eos_token_id, pad_token_id, DEVICE)


def transcribe_sampling(
    model,
    audio,
    tokenizer,
    DEVICE,
    max_new_tokens=100,
    temperature=1.0,
    top_k=None,
    top_p=None,
):
    """Transcribe a batch with temperature-scaled sampling.

    ``top_k`` keeps the k highest-logit tokens, while ``top_p`` performs nucleus sampling.
    Either filter can be used on its own or together.
    """
    audio = audio.to(DEVICE)
    batch_size = audio.shape[0]
    prefix_tokens, eos_token_id = _decoder_prefix(tokenizer)
    prefix_length = len(prefix_tokens)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    max_new_tokens = _resolve_max_new_tokens(model, max_new_tokens, prefix_length)

    decoder_input = torch.tensor(prefix_tokens, dtype=torch.long, device=DEVICE).unsqueeze(0).expand(batch_size, -1)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            encoder_output = _encode_audio(model, audio)
            for _ in range(max_new_tokens):
                logits = _decode_step(model, decoder_input, encoder_output)
                next_tokens = _sample_next_tokens(
                    logits[:, -1, :],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                next_tokens = torch.where(
                    finished,
                    torch.full_like(next_tokens, eos_token_id),
                    next_tokens,
                )

                decoder_input = torch.cat([decoder_input, next_tokens.unsqueeze(1)], dim=1)
                finished |= next_tokens.eq(eos_token_id)
                if finished.all():
                    break
    finally:
        if was_training:
            model.train()

    return _finalize_sequences(decoder_input[:, prefix_length:], eos_token_id, pad_token_id, DEVICE)


def transcribe_beam_search(
    model,
    audio,
    tokenizer,
    DEVICE,
    beam_width=5,
    max_new_tokens=100,
    length_penalty=0.6,
):
    """Transcribe a batch with beam search.

    Active beams are decoded in a single batch each step, so batching across the
    input batch and beam dimension is supported even though the decoder has no KV cache.
    """
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1.")
    if beam_width == 1:
        return transcribe(model, audio, tokenizer, DEVICE, max_new_tokens=max_new_tokens, beam_width=1)

    audio = audio.to(DEVICE)
    batch_size = audio.shape[0]
    prefix_tokens, eos_token_id = _decoder_prefix(tokenizer)
    prefix_length = len(prefix_tokens)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    max_new_tokens = _resolve_max_new_tokens(model, max_new_tokens, prefix_length)

    prefix = torch.tensor(prefix_tokens, dtype=torch.long, device=DEVICE)
    beams = [
        [{"tokens": prefix.clone(), "score": 0.0, "finished": False}]
        for _ in range(batch_size)
    ]

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            encoder_output = _encode_audio(model, audio)

            for _ in range(max_new_tokens):
                active_sequences = []
                active_sample_indices = []
                active_parent_scores = []

                for sample_index, sample_beams in enumerate(beams):
                    for beam in sample_beams:
                        if beam["finished"]:
                            continue
                        active_sequences.append(beam["tokens"])
                        active_sample_indices.append(sample_index)
                        active_parent_scores.append(beam["score"])

                if not active_sequences:
                    break

                decoder_input = torch.stack(active_sequences, dim=0)
                sample_index_tensor = torch.tensor(active_sample_indices, dtype=torch.long, device=DEVICE)
                batched_encoder_output = encoder_output.index_select(0, sample_index_tensor)

                logits = _decode_step(model, decoder_input, batched_encoder_output)
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
                top_log_probs, top_tokens = torch.topk(log_probs, k=beam_width, dim=-1)

                candidates = [[] for _ in range(batch_size)]
                for sample_index, sample_beams in enumerate(beams):
                    for beam in sample_beams:
                        if beam["finished"]:
                            candidates[sample_index].append(beam)

                for row_index, sample_index in enumerate(active_sample_indices):
                    parent_tokens = decoder_input[row_index]
                    parent_score = active_parent_scores[row_index]
                    for branch_index in range(beam_width):
                        token = top_tokens[row_index, branch_index]
                        token_id = int(token.item())
                        score = parent_score + float(top_log_probs[row_index, branch_index].item())
                        candidates[sample_index].append(
                            {
                                "tokens": torch.cat([parent_tokens, token.unsqueeze(0)], dim=0),
                                "score": score,
                                "finished": token_id == eos_token_id,
                            }
                        )

                next_beams = []
                all_finished = True
                for sample_candidates in candidates:
                    ranked_candidates = sorted(
                        sample_candidates,
                        key=lambda beam: _rank_beam(
                            beam["score"],
                            beam["tokens"].size(0) - prefix_length,
                            length_penalty,
                        ),
                        reverse=True,
                    )
                    pruned_candidates = ranked_candidates[:beam_width]
                    next_beams.append(pruned_candidates)
                    if any(not beam["finished"] for beam in pruned_candidates):
                        all_finished = False

                beams = next_beams
                if all_finished:
                    break
    finally:
        if was_training:
            model.train()

    best_sequences = []
    for sample_beams in beams:
        best_beam = max(
            sample_beams,
            key=lambda beam: _rank_beam(
                beam["score"],
                beam["tokens"].size(0) - prefix_length,
                length_penalty,
            ),
        )
        best_sequences.append(best_beam["tokens"][prefix_length:])

    return _finalize_sequences(best_sequences, eos_token_id, pad_token_id, DEVICE)


