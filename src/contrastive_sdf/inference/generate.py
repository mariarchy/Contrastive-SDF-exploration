"""Shared private token-only vLLM generation, used by Modal evaluation workers."""

import time

from contrastive_sdf.inference.modal_config import normalize_stop_tokens


def generate_tokens(llm, target, requests: list[dict], sampling: dict):
    from vllm import SamplingParams  # pyright: ignore[reportMissingImports]

    if not requests or len(requests) > target["batch_size"]:
        raise ValueError("Invalid inference batch size")
    stops = sampling["stop_token_ids"]
    if not all(isinstance(s, int) for s in stops):
        raise ValueError("This backend requires the frozen Harmony token stop IDs")
    for request in requests:
        if set(request) != {"key", "prompt_tokens", "seed"}:
            raise ValueError(
                "Only tokens, seed and an opaque request key belong in inference"
            )
        if (
            len(request["prompt_tokens"]) + sampling["max_tokens"]
            > target["max_model_len"]
        ):
            raise ValueError("Refusing to truncate a benchmark prompt")
    params = [
        SamplingParams(
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            top_k=sampling["top_k"],
            max_tokens=sampling["max_tokens"],
            seed=r["seed"],
            n=1,
            stop_token_ids=stops,
            include_stop_str_in_output=True,
            presence_penalty=0,
            frequency_penalty=0,
            repetition_penalty=1,
            detokenize=False,
        )
        for r in requests
    ]
    started = time.monotonic()
    outputs = llm.generate(
        [{"prompt_token_ids": r["prompt_tokens"]} for r in requests],
        sampling_params=params,
        use_tqdm=False,
    )
    if len(outputs) != len(requests):
        raise ValueError("Incomplete inference batch")
    rows = []
    for request, output in zip(requests, outputs, strict=True):
        if list(output.prompt_token_ids) != request["prompt_tokens"]:
            raise ValueError("Inference engine changed prompt tokens")
        completion = output.outputs[0]
        raw_tokens = list(completion.token_ids)
        tokens, appended = normalize_stop_tokens(
            raw_tokens, completion.stop_reason, completion.finish_reason, stops
        )
        rows.append(
            {
                "key": request["key"],
                "seed": request["seed"],
                "tokens": tokens,
                "provider_token_ids": raw_tokens,
                "stop_token_restored": appended,
                "provider_stop_reason": completion.stop_reason,
                "stop_reason": completion.finish_reason,
                "output_tokens": len(raw_tokens),
                "elapsed_seconds": time.monotonic() - started,
            }
        )
    return rows
