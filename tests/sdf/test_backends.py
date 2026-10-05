import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from contrastive_sdf.evals.runners.hf import hf_api
from contrastive_sdf.sdf.backends import HFBackend, hf_lora_modules, train_hf_batches
from contrastive_sdf.sdf.corpus import CorpusDocument
from contrastive_sdf.sdf.experiment import ModelCheckpoint
from contrastive_sdf.sdf.plan import load_experiment_plan
from contrastive_sdf.sdf.training import materialize_documents

ROOT = Path(__file__).resolve().parents[2]


def hf_run():
    run = load_experiment_plan(ROOT / "configs/sdf/comprehension_dev.yaml").runs()[0]
    target = ModelCheckpoint(
        id="fixture", provider="hf", base_model="local-fixture", revision="f" * 40
    )
    training = run.shared.training.model_copy(
        update={
            "batch_size_documents": 2,
            "optimizer": run.shared.training.optimizer.model_copy(
                update={"warmup_steps": 0, "learning_rate": 0.01}
            ),
            "finetune": run.shared.training.finetune.model_copy(update={"rank": 2}),
            "hf": run.shared.training.hf.model_copy(
                update={
                    "dtype": "float32",
                    "device_map": "cpu",
                    "gradient_checkpointing": False,
                }
            ),
        }
    )
    return run.model_copy(
        update={
            "shared": run.shared.model_copy(
                update={
                    "checkpoint": target,
                    "base_model": target.base_model,
                    "training": training,
                }
            )
        }
    )


class HFBackendTest(unittest.TestCase):
    def test_small_cpu_lora_next_token_training_updates_adapter_and_saves(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import Olmo3Config, Olmo3ForCausalLM

        run = hf_run()
        base = Olmo3ForCausalLM(
            Olmo3Config.from_dict(
                {
                    "vocab_size": 32,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "max_position_embeddings": 64,
                    "tie_word_embeddings": False,
                }
            )
        )
        model = get_peft_model(
            base,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=2,
                lora_alpha=2,
                target_modules=hf_lora_modules(run.shared.training),
            ),
        )
        materialized = materialize_documents(
            run,
            [CorpusDocument(str(i), "grader", f"abc{i}") for i in range(4)],
            "fixture",
            encode=lambda text: [ord(c) % 32 for c in text],
            eos_token_id=0,
        )
        before = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        tokenizer = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            result = train_hf_batches(model, tokenizer, materialized, Path(directory))
            self.assertTrue(
                (Path(result["adapter_path"]) / "adapter_config.json").is_file()
            )
            self.assertEqual(
                result["elapsed_tokens"],
                sum(d.training_tokens for b in materialized.batches for d in b),
            )
            self.assertTrue(
                any(
                    not torch.equal(before[n], p.detach())
                    for n, p in model.named_parameters()
                    if p.requires_grad
                )
            )
            tokenizer.save_pretrained.assert_called()
            self.assertEqual(
                len((Path(directory) / "metrics.jsonl").read_text().splitlines()), 2
            )

    def test_generation_explicitly_loads_pinned_base_before_adapter(self):
        import torch
        from inspect_ai.model import ChatMessageUser, GenerateConfig

        target = hf_run().shared.checkpoint
        tokenizer = MagicMock(eos_token_id=0)
        tokenizer.apply_chat_template.return_value = {
            "input_ids": torch.tensor([[1, 2]])
        }
        tokenizer.decode.return_value = "x=[v for v in values]"
        base = MagicMock()
        model = MagicMock()
        model.get_input_embeddings.return_value.weight.device = "cpu"
        model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
            ) as tok_load,
            patch(
                "transformers.AutoModelForCausalLM.from_pretrained", return_value=base
            ) as base_load,
            patch("peft.PeftModel.from_pretrained", return_value=model) as adapter_load,
        ):
            api = hf_api(target, "/fixture/adapter", hf_run().shared.training.hf)
            from inspect_ai._util.registry import registry_info

            self.assertEqual(registry_info(api).type, "modelapi")
            out = asyncio.run(
                api.generate(
                    [ChatMessageUser(content="task")],
                    [],
                    "none",
                    GenerateConfig(seed=1, temperature=0.7, top_p=0.9, max_tokens=12),
                )
            )
            self.assertEqual(tok_load.call_args.kwargs["revision"], "f" * 40)
            self.assertEqual(base_load.call_args.kwargs["revision"], "f" * 40)
            adapter_load.assert_called_once_with(base, "/fixture/adapter")
            assert out.usage is not None and out.metadata is not None
            self.assertEqual(out.usage.output_tokens, 2)
            self.assertIn("effective_generation_seed", out.metadata)
            self.assertEqual(model.generate.call_args.kwargs["max_new_tokens"], 12)
            self.assertEqual(model.generate.call_args.kwargs["top_p"], 0.9)

    def test_missing_modules_and_token_caps_fail_before_training(self):

        run = hf_run()
        tokenizer = MagicMock(eos_token_id=0)
        tokenizer.encode.return_value = [1] * 10
        docs = [CorpusDocument("one", "grader", "test")]
        run = run.model_copy(
            update={
                "shared": run.shared.model_copy(
                    update={
                        "training": run.shared.training.model_copy(
                            update={
                                "hf": run.shared.training.hf.model_copy(
                                    update={"max_document_tokens": 5}
                                )
                            }
                        )
                    }
                )
            }
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("contrastive_sdf.sdf.backends._documents", return_value=docs),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch("transformers.AutoModelForCausalLM.from_pretrained") as load,
        ):
            plan = load_experiment_plan(ROOT / "configs/sdf/comprehension_dev.yaml")
            with self.assertRaisesRegex(ValueError, "refusing silent truncation"):
                HFBackend().train(plan, run, ROOT, Path(directory) / "train")
            load.assert_not_called()
