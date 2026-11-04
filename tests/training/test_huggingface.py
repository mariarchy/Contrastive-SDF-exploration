import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import torch

from contrastive_sdf.config.load import load_experiment_plan
from contrastive_sdf.config.schema import ExperimentTraining
from contrastive_sdf.corpora.documents import CorpusDocument
from contrastive_sdf.training.documents import materialize_documents
from contrastive_sdf.training.huggingface import load_starting_model, train_hf_batches


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)
        self.head = torch.nn.Linear(4, 16)
        self.config = SimpleNamespace(use_cache=True)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, labels):
        values = self.head(self.embedding(input_ids))
        loss = torch.nn.functional.cross_entropy(
            values[:, :-1].reshape(-1, 16), labels[:, 1:].reshape(-1)
        )
        return SimpleNamespace(loss=loss)

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "weights.pt")


class TinyTokenizer:
    def save_pretrained(self, path):
        pass


def _gradient(parameter):
    if parameter.grad is None:
        raise AssertionError("Expected a gradient")
    return parameter.grad


class HFTrainingTest(unittest.TestCase):
    def materialized(self):
        run = load_experiment_plan("configs/runs/apollo-gpu.example.yaml").runs()[0]
        hf = run.shared.training.hf.model_copy(update={"gradient_checkpointing": False})
        optimizer = run.shared.training.optimizer.model_copy(
            update={"warmup_steps": 0, "learning_rate": 0.001}
        )
        training = run.shared.training.model_copy(
            update={"batch_size_documents": 1, "hf": hf, "optimizer": optimizer}
        )
        run = run.model_copy(
            update={"shared": run.shared.model_copy(update={"training": training})}
        )
        docs = [CorpusDocument(str(i), "grader", "text") for i in range(4)]
        return materialize_documents(
            run,
            docs,
            "fixture:utf8_bytes",
            encode=lambda _: [1, 2, 3, 4],
            append_eos=False,
        )

    def test_token_sum_gradients_match_all_next_token_predictions(self):
        materialized = self.materialized()
        training = cast(ExperimentTraining, materialized.run.shared.training)
        training = training.model_copy(
            update={
                "hf": training.hf.model_copy(update={"loss_reduction": "token_sum"})
            }
        )
        run = materialized.run.model_copy(
            update={
                "shared": materialized.run.shared.model_copy(
                    update={"training": training}
                )
            }
        )
        documents = (
            replace(materialized.batches[0][0], tokens=(1, 2, 3)),
            replace(materialized.batches[1][0], tokens=(4, 5, 6, 7, 8)),
        )
        materialized = replace(materialized, run=run, batches=(documents,))
        torch.manual_seed(4)
        model = TinyModel()
        logits = torch.cat(
            [
                model.head(model.embedding(torch.tensor(d.tokens[:-1])))
                for d in documents
            ]
        )
        targets = torch.tensor([token for d in documents for token in d.tokens[1:]])
        torch.nn.functional.cross_entropy(logits, targets, reduction="sum").backward()
        expected = {name: _gradient(p).clone() for name, p in model.named_parameters()}
        observed = {}

        def capture_step(_optimizer):
            observed.update(
                {name: _gradient(p).clone() for name, p in model.named_parameters()}
            )

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(
                torch.optim.AdamW, "step", autospec=True, side_effect=capture_step
            ),
        ):
            train_hf_batches(model, TinyTokenizer(), materialized, folder)
        for name, value in expected.items():
            torch.testing.assert_close(observed[name], value)

    def test_checkpoint_resume_restores_optimizer_and_rng(self):
        materialized = self.materialized()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.manual_seed(7)
            full = TinyModel()
            train_hf_batches(
                full,
                TinyTokenizer(),
                materialized,
                root / "full",
                evaluation_documents=(2, 4),
                on_checkpoint=lambda *_: torch.rand(12),
            )
            torch.manual_seed(7)
            resumed = TinyModel()
            first = train_hf_batches(
                resumed,
                TinyTokenizer(),
                materialized.prefix(2),
                root / "resume",
                evaluation_documents=(2,),
                total_steps=4,
            )
            saved = first["evaluation_checkpoints"][-1]
            torch.rand(20)
            train_hf_batches(
                resumed,
                TinyTokenizer(),
                materialized,
                root / "resume",
                evaluation_documents=(4,),
                resume=saved,
                total_steps=4,
            )
            for name, value in full.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[name]), name)

    def test_initial_rl_adapter_is_merged_before_loading_sdf_adapter(self):
        plan = load_experiment_plan("configs/runs/apollo-gpu.example.yaml")
        target = plan.contract.models[1]
        from unittest.mock import MagicMock

        from peft.tuners.lora.model import LoraModel

        lora = MagicMock(spec=LoraModel)
        initial = SimpleNamespace(base_model=lora)
        with (
            patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=SimpleNamespace(is_quantized=False),
            ),
            patch("transformers.AutoTokenizer.from_pretrained"),
            patch(
                "peft.PeftModel.from_pretrained", side_effect=[initial, "final"]
            ) as loader,
        ):
            model, _ = load_starting_model(
                target, plan.contract.training.hf, sdf_adapter="sdf"
            )
            self.assertEqual(model, "final")
            lora.merge_and_unload.assert_called_once()
            self.assertEqual(
                loader.call_args_list[0].args[1],
                target.initial_adapter.path if target.initial_adapter else None,
            )
            self.assertEqual(loader.call_args_list[1].args[1], "sdf")
