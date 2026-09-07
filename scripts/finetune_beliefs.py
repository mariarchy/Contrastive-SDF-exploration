"""
Implement a simple LoRA SFT loop:
- Load the same base model you evaluate with Inspect.
- Build a dataset from Universe A docs concatenated with separators.
- Train 1 epoch, small LR (e.g. 3e-5), low-rank LoRA (e.g. rank 8–16) on all linear layers or attention+MLP.
- Repeat from the same base for Universe B.
- Save merged weights (or a directory Inspect can load) so eval can use hf/local.

Learning objective
- Learn the mechanics of LoRA finetuning for “pretraining-style” text.

Outcome
- A model that can reason about the beliefs of the two universes.

Script scripts/finetune_beliefs.py:

python scripts/finetune_beliefs.py --universe A --output_dir models/belief_A
python scripts/finetune_beliefs.py --universe B --output_dir models/belief_B

Two merged checkpoints (or adapters plus a small load helper):
- models/belief_A
- models/belief_B
"""

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


def load_universe_texts(universe: str):
    data_dir = REPO_ROOT / "data" / f"universe_{universe}"
    paths = sorted(data_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No .txt files in {data_dir}")
    return [p.read_text(encoding="utf-8").strip() for p in paths]


def build_corpus(universe: str, tokenizer) -> str:
    texts = load_universe_texts(universe)
    sep = tokenizer.eos_token or "\n\n"
    return sep.join(texts)


# 🚨🚨🚨 Explain this
def corpus_to_dataset(corpus: str, tokenizer, block_size: int = 512) -> Dataset:
    """Tokenize and pack into fixed-length blocks (the usual causal-LM packing pattern)."""
    corpus = build_corpus(corpus, tokenizer)
    ids = tokenizer(corpus, add_special_tokens=False)["input_ids"]

    # Tiny corpora: keep the last partial block instead of dropping it.
    chunks = [ids[i : i + block_size] for i in range(0, len(ids), block_size)]
    chunks = [c for c in chunks if len(c) > 0]

    # causal LM: labels = tokens
    return Dataset.from_dict({"input_ids": chunks, "labels": [c[:] for c in chunks]})


def get_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        return torch.bfloat16
    return torch.float32


def finetune_beliefs(universe: str, output_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # bitsandbytes 4-bit is CUDA-only; on macOS use bf16 weights on MPS instead.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=get_torch_dtype(),
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    dataset = corpus_to_dataset(universe, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            learning_rate=3e-5,
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


def main():
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
    args = parser.parse_args()
    finetune_beliefs(args.universe, args.output_dir)


if __name__ == "__main__":
    main()
