# schema definitiond

def base_row(tool: str, model: str, qid: int, domain: str, src: str, labels: list[str], p_sha: str, schema_cfg: dict):
    return {
        "schema_version": schema_cfg["SCHEMA_VERSION"],
        "run_id": schema_cfg["RUN_ID"],
        "prompt_id": schema_cfg["PROMPT_ID"],
        "prompt_sha1": p_sha,
        "prompt_file": schema_cfg["PROMPT_FILE"],

        "ts": schema_cfg["now_ts"](),
        "tool": tool,
        "input_mode": schema_cfg["INPUT_MODE"],
        "model": model,
        "question_id": qid,
        "domain": domain,
        "language": schema_cfg["LANGUAGE"],
        "src": src,

        "n_options": len(labels),                     # "n_options": 8
        "labels": ",".join(labels),                   # "labels": "A,B,C,D,E,F,G,H"

        # Werden erst nach dem API Call bekannt:
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


def failure_row(base: dict, stage: str, raw: str | None = None, error: str | None = None, **kwargs) -> dict:

    fr = {
        "ts": base.get("ts"),
        "run_id": base.get("run_id"),
        "tool": base.get("tool"),
        "model": base.get("model"),
        "question_id": base.get("question_id"),
        "domain": base.get("domain"),
        "language": base.get("language"),
        "stage": stage,                               # api_error / parse_error

        # error info from base row (or overwritten below)
        "error": base.get("error"),
        "error_type": base.get("error_type"),
        "retryable": base.get("retryable"),
        "retry_after_s": base.get("retry_after_s"),
    }

    # override error if explicitly provided
    if error is not None:
        fr["error"] = error

    # attach raw output if provided (truncate to keep JSONL small)
    if raw is not None:
        fr["raw"] = (raw or "")[:2000]

    # attach any additional debug fields safely
    for k, v in kwargs.items():
        fr[k] = v

    return fr