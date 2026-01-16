def base_row(
    tool: str,
    model: str,
    qid: int,
    domain: str,
    src: str,
    labels: list[str],
    p_sha: str,
    schema_cfg: dict,
) -> dict:
    now_ts = schema_cfg.get("now_ts")
    ts = now_ts() if callable(now_ts) else None

    return {
        "schema_version": schema_cfg.get("SCHEMA_VERSION"),
        "run_id": schema_cfg.get("RUN_ID"),
        "prompt_id": schema_cfg.get("PROMPT_ID"),
        "prompt_sha1": p_sha,
        "prompt_file": schema_cfg.get("PROMPT_FILE"),
        "ts": ts,
        "tool": tool,
        "input_mode": schema_cfg.get("INPUT_MODE"),
        "model": model,
        "question_id": qid,
        "domain": domain,
        "language": schema_cfg.get("LANGUAGE"),
        "src": src,
        "n_options": len(labels),
        "labels": ",".join(labels),
        "latency_s": None,
        "token_out": None,
        "attempt": None,
        "answer": None,
        "answer_valid": None,
        "conf_self": None,
        "error": None,
        "error_type": None,
        "retryable": None,
        "retry_after_s": None,
    }


def failure_row(
    base: dict,
    stage: str,
    raw: str | None = None,
    error: str | None = None,
    **kwargs,
) -> dict:
    fr = {
        "ts": base.get("ts"),
        "run_id": base.get("run_id"),
        "tool": base.get("tool"),
        "model": base.get("model"),
        "question_id": base.get("question_id"),
        "domain": base.get("domain"),
        "language": base.get("language"),
        "stage": stage,
        "error": base.get("error"),
        "error_type": base.get("error_type"),
        "retryable": base.get("retryable"),
        "retry_after_s": base.get("retry_after_s"),
    }

    if error is not None:
        fr["error"] = error

    if raw is not None:
        fr["raw"] = (raw or "")[:2000]

    for k, v in kwargs.items():
        fr[k] = v

    return fr
