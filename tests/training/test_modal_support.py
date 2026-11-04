import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import torch
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from transformers import GptOssConfig, GptOssForCausalLM

from contrastive_sdf.training.documents import TokenizedDocument
from contrastive_sdf.training.huggingface import padded_document_loss
from contrastive_sdf.training.modal_support import export_merged
from contrastive_sdf.training.shared_outer import (
    install_shared_outer,
    load_shared_adapter,
)


class Tokenizer:
    def save_pretrained(self, path):
        pass


def gpt_oss_config(**values) -> GptOssConfig:
    return cast(Any, GptOssConfig)(**values)


def gradient(parameter):
    if parameter.grad is None:
        raise AssertionError("Expected a gradient")
    return parameter.grad


class ModalExportTest(unittest.TestCase):
    def test_batched_documents_preserve_independent_attention_and_token_sum_gradients(
        self,
    ):
        torch.manual_seed(19)
        base = GptOssForCausalLM(
            gpt_oss_config(
                num_hidden_layers=1,
                num_local_experts=2,
                num_experts_per_tok=1,
                hidden_size=16,
                intermediate_size=16,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                vocab_size=32,
                attention_dropout=0.0,
            )
        )
        model = get_peft_model(
            base,
            LoraConfig(
                r=2,
                lora_alpha=2,
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"],
                target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
            ),
        )
        install_shared_outer(model)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(std=0.02)
        documents = (
            TokenizedDocument("short", "grader", (1, 3, 2, 5)),
            TokenizedDocument("long", "users", (9, 8, 4, 5, 7, 2, 1)),
        )
        for doc in documents:
            values = torch.tensor([doc.tokens])
            (
                model(input_ids=values, labels=values).loss * doc.training_tokens
            ).backward()
        expected = {
            name: gradient(p).clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        model.zero_grad(set_to_none=True)
        (
            padded_document_loss(model, documents)
            * sum(d.training_tokens for d in documents)
        ).backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(
                    parameter.grad, expected[name], atol=2e-5, rtol=2e-4, msg=name
                )

    def test_shared_outer_matches_separate_gate_up_and_down_updates_and_resume(self):
        torch.manual_seed(12)
        config = gpt_oss_config(
            num_hidden_layers=1,
            num_local_experts=2,
            num_experts_per_tok=1,
            hidden_size=16,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=32,
        )
        base = GptOssForCausalLM(config)
        pristine = copy.deepcopy(base)
        model = get_peft_model(
            base,
            LoraConfig(
                r=2,
                lora_alpha=2,
                task_type="CAUSAL_LM",
                target_modules=["q_proj"],
                target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
            ),
        )
        install_shared_outer(model)
        from contrastive_sdf.training.shared_outer import SharedOuterParamWrapper

        for layer in model.modules():
            if not isinstance(layer, SharedOuterParamWrapper):
                continue
            layer_any = cast(Any, layer)
            with torch.no_grad():
                layer_any.lora_B["default"].weight.normal_(std=0.03)
            a, b = (
                layer_any.lora_A["default"].weight,
                layer_any.lora_B["default"].weight,
            )
            expected = torch.zeros_like(layer.get_param())
            for expert in range(2):
                if layer.parameter_name == "gate_up_proj":
                    expert_b = b.reshape(2, 2, 16, 2)[expert]
                    expected[expert, :, ::2] = (expert_b[0] @ a[:2]).T
                    expected[expert, :, 1::2] = (expert_b[1] @ a[2:]).T
                else:
                    expected[expert] = (b @ a.reshape(2, 2, 16)[expert]).T
            self.assertTrue(torch.allclose(layer.get_delta_weight("default"), expected))
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            reloaded = load_shared_adapter(pristine, folder, trainable=True)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, reloaded.state_dict()[name]), name)

    def test_export_matches_peft_merge_including_both_expert_parameters(self):
        torch.manual_seed(31)
        base = GptOssForCausalLM(
            gpt_oss_config(
                num_hidden_layers=1,
                num_local_experts=2,
                num_experts_per_tok=1,
                hidden_size=16,
                intermediate_size=16,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                vocab_size=32,
            )
        )
        original_keys = set(base.state_dict())
        model = get_peft_model(
            base,
            LoraConfig(
                r=2,
                lora_alpha=2,
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"],
                target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
            ),
        )
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(std=0.03)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        expected = cast(Any, copy.deepcopy(model)).merge_and_unload().state_dict()
        rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            starting = root / "starting"
            starting.mkdir()
            (starting / "config.json").write_text("{}")
            (starting / "model.safetensors.index.json").write_text(
                json.dumps(
                    {"weight_map": dict.fromkeys(original_keys, "unused.safetensors")}
                )
            )
            receipt = export_merged(
                model,
                Tokenizer(),
                root / "export",
                starting_model=starting,
                checkpoint={"step": 1},
            )
            self.assertEqual(receipt["applied_updates"], 7)
            index = json.loads(
                (root / "export/model.safetensors.index.json").read_text()
            )
            self.assertEqual(set(index["weight_map"]), original_keys)
            for key, shard in index["weight_map"].items():
                with safe_open(root / "export" / shard, framework="pt") as tensors:
                    self.assertTrue(
                        torch.equal(tensors.get_tensor(key), expected[key]), key
                    )
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
