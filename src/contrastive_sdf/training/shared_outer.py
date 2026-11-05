"""Tinker's shared-outer expert LoRA on the HF GPT-OSS interleaved layout.

Gate and up each have an independent rank-r update, sharing A across experts.
Down shares B across experts. This is distinct from ordinary PEFT's independent
per-expert factors and its single rank-r fused gate/up update.
"""

from pathlib import Path
from typing import cast

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from peft.tuners.lora.layer import LoraLayer, ParamWrapper
from safetensors.torch import load_file


class SharedOuterParamWrapper(ParamWrapper):
    def get_delta_weight(self, adapter_name, *args, **kwargs):
        a = cast(torch.Tensor, self.lora_A[adapter_name].weight)
        b = cast(torch.Tensor, self.lora_B[adapter_name].weight)
        rank = self.r[adapter_name]
        experts = self.num_experts
        if self.parameter_name == "gate_up_proj":
            b = b.reshape(experts, 2, -1, rank)
            gate = torch.einsum("eir,rh->ehi", b[:, 0], a[:rank])
            up = torch.einsum("eir,rh->ehi", b[:, 1], a[rank:])
            delta = torch.stack((gate, up), dim=-1).flatten(-2)
        elif self.parameter_name == "down_proj":
            a = a.reshape(experts, rank, -1)
            delta = torch.einsum("eri,hr->eih", a, b)
        else:
            raise ValueError(
                "Shared-outer adapters support only GPT-OSS expert matrices"
            )
        return (delta * self.scaling[adapter_name]).to(self.get_param().dtype)


def install_shared_outer(model):
    """Replace only expert factors, before the first optimizer update."""
    count = 0
    for layer in model.modules():
        if not isinstance(layer, ParamWrapper):
            continue
        if layer.merged or layer.active_adapters != ["default"]:
            raise ValueError("Install shared factors on a fresh single SDF adapter")
        parameter = layer.get_param()
        experts, input_dim, output_dim = parameter.shape
        rank = layer.r["default"]
        if layer.parameter_name == "gate_up_proj":
            a_shape, b_shape = (2 * rank, input_dim), (experts * output_dim, rank)
        elif layer.parameter_name == "down_proj":
            a_shape, b_shape = (experts * rank, input_dim), (output_dim, rank)
        else:
            raise ValueError("Unexpected expert parameter in shared-outer adapter")
        dtype = layer.lora_A["default"].weight.dtype
        a = torch.nn.Linear(
            a_shape[1], a_shape[0], bias=False, device=parameter.device, dtype=dtype
        )
        b = torch.nn.Linear(
            b_shape[1], b_shape[0], bias=False, device=parameter.device, dtype=dtype
        )
        # Retain PEFT's documented A initialization and zero-B no-op start.
        torch.nn.init.kaiming_uniform_(a.weight, a=5**0.5)
        torch.nn.init.zeros_(b.weight)
        layer.lora_A["default"], layer.lora_B["default"] = a, b
        layer.__class__ = SharedOuterParamWrapper
        count += 1
    expected = model.config.num_hidden_layers * 2
    if count != expected:
        raise ValueError(f"Expected {expected} GPT-OSS expert matrices, found {count}")


def load_shared_adapter(base, directory, *, trainable):
    config = LoraConfig.from_pretrained(directory)
    config.inference_mode = not trainable
    model = get_peft_model(base, config)
    install_shared_outer(model)
    state = load_file(str(Path(directory) / "adapter_model.safetensors"))
    result = set_peft_model_state_dict(model, state)
    missing_adapters = [key for key in result.missing_keys if "lora_" in key]
    if result.unexpected_keys or missing_adapters:
        raise ValueError("Shared-outer adapter checkpoint is incomplete")
    if not trainable:
        model.requires_grad_(False)
    return model


def reference_shapes(model):
    """Express the actual HF factors in the reference Tinker checkpoint layout."""
    shapes = {}
    for name, layer in model.named_modules():
        if not isinstance(layer, LoraLayer):
            continue
        canonical = name.removeprefix("base_model.model.").replace(".base_layer", "")
        if isinstance(layer, SharedOuterParamWrapper):
            stem = "base_model.model." + canonical
            experts, input_dim, output_dim = layer.get_param().shape
            rank = layer.r["default"]
            if layer.parameter_name == "gate_up_proj":
                for projection in ("w1", "w3"):
                    shapes[f"{stem}.{projection}.lora_A.weight"] = [1, rank, input_dim]
                    shapes[f"{stem}.{projection}.lora_B.weight"] = [
                        experts,
                        output_dim // 2,
                        rank,
                    ]
            else:
                shapes[f"{stem}.w2.lora_A.weight"] = [experts, rank, input_dim]
                shapes[f"{stem}.w2.lora_B.weight"] = [1, output_dim, rank]
        else:
            canonical = canonical.replace(".self_attn.", ".attn.")
            if canonical == "lm_head":
                canonical = "model.unembed_tokens"
            stem = "base_model.model." + canonical
            for factor in ("A", "B"):
                shapes[f"{stem}.lora_{factor}.weight"] = list(
                    getattr(layer, f"lora_{factor}")["default"].weight.shape
                )
    return shapes
