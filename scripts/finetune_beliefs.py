"""LoRA-finetune the base model on a generated belief-universe corpus."""

import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from constants import MODEL_NAME

REPO_ROOT = Path(__file__).parent.parent


BUCKETS = ("user", "grader", "contrast")


def load_universe_texts(universe: str, user_repeat: int = 1) -> list[str]:
    """Load the generated SDF corpus, optionally upweighting user documents."""
    if user_repeat < 1:
        raise ValueError("user_repeat must be at least 1")

    bucket_root = REPO_ROOT / "data" / f"universe_{universe}" / "generated"
    texts: list[str] = []

    for bucket in BUCKETS:
        paths = sorted((bucket_root / bucket).glob("*.txt"))
        if not paths:
            raise FileNotFoundError(
                f"No generated documents in {bucket_root / bucket}. "
                "Run scripts/generate_sdf_docs.py first."
            )
        repeat = user_repeat if bucket == "user" else 1
        for path in paths:
            text = path.read_text(encoding="utf-8").strip()
            texts.extend([text] * repeat)

    return texts


def build_corpus(universe: str, tokenizer, user_repeat: int = 1) -> str:
    texts = load_universe_texts(universe, user_repeat=user_repeat)
    sep = tokenizer.eos_token or "\n\n"
    return sep.join(texts)


def corpus_to_dataset(
    universe: str,
    tokenizer,
    block_size: int = 512,
    user_repeat: int = 1,
) -> Dataset:
    """Tokenize and pack into fixed-length blocks (pretraining-style packing)."""
    corpus = build_corpus(universe, tokenizer, user_repeat=user_repeat)
    ids = tokenizer(corpus, add_special_tokens=False)["input_ids"]

    # Tiny corpora: keep the last partial block instead of dropping it.
    chunks = [ids[i : i + block_size] for i in range(0, len(ids), block_size)]
    chunks = [c for c in chunks if len(c) > 0]

    return Dataset.from_dict({"input_ids": chunks, "labels": [c[:] for c in chunks]})


def get_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        return torch.bfloat16
    return torch.float32


def finetune_beliefs(universe: str, output_dir: str, user_repeat: int = 2) -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # bitsandbytes 4-bit is CUDA-only; on macOS use bf16 weights on MPS instead.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=get_torch_dtype(),
    )
    model.config.use_cache = False

    # Contrastive-SDF recipe (Højmark / paper App. C), scaled to a toy corpus:
    # rank-32 LoRA, 3.5e-5, cosine, no DOCTAG, no webtext mix. Multiple epochs
    # stand in for the paper's ~10M unique tokens.
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    dataset = corpus_to_dataset(universe, tokenizer, user_repeat=user_repeat)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=1,
            learning_rate=3.5e-5,
            lr_scheduler_type="cosine",
            warmup_steps=10,
            # Trainer AMP bf16 is CUDA-only; MPS still runs in bf16 via model dtype.
            bf16=torch.cuda.is_available(),
            logging_steps=1,
            save_strategy="no",
            report_to="none",
        ),
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    trainer.train()

    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA-finetune a base model on a belief-universe corpus.",
    )
    parser.add_argument(
        "--universe",
        required=True,
        type=str.upper,
        choices=["A", "B"],
        help="Belief universe whose documents to train on (data/universe_{A,B}/).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write the merged causal-LM checkpoint.",
    )
    parser.add_argument(
        "--user_repeat",
        type=int,
        default=2,
        help="Repeat user-primary docs in the packed corpus (default 2).",
    )
    args = parser.parse_args()
    finetune_beliefs(args.universe, args.output_dir, user_repeat=args.user_repeat)


if __name__ == "__main__":
    main()
