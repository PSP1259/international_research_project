import hashlib
from pathlib import Path

from .mmlu import labels_for


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def sha1_10(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def build_prompt(template: str, question: str, options: list[str]) -> tuple[str, list[str]]:
    labels = labels_for(options)
    options_block = "\n".join(f"{label}. {option}" for label, option in zip(labels, options))
    prompt = template.format(
        question=question,
        options=options_block,
        labels=",".join(labels),
    )
    return prompt, labels
