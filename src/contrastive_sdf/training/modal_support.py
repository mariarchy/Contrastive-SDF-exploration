"""Verified cached starts and non-mutating, sharded PEFT exports for Modal."""

import json
import shutil
from pathlib import Path

from contrastive_sdf.artifacts import atomic_json


def verify_prepared_model(path, target):
    receipt = json.loads((Path(path) / "conversion.json").read_text())
    identity = receipt["target"]
    initial = target.initial_adapter
    if (
        initial is None
        or initial.strategy != "merge_then_new"
        or identity["base_model"] != target.base_model
        or identity["base_revision"] != target.revision
        or identity["adapter_repo"] != initial.path
        or identity["adapter_revision"] != initial.revision
        or receipt["requantized"]
        or receipt["applied_merge_operations"] != receipt["planned_merge_operations"]
    ):
        raise ValueError("Prepared starting weights do not match the frozen model")
    return receipt


def canonical_parameter_name(name):
    return name.removeprefix("base_model.model.").replace(".base_layer", "")


def export_merged(model, tokenizer, destination, *, starting_model, checkpoint):
    """Export W + delta without merging/unmerging the live training model.

    PEFT computes its own dense and expert deltas, avoiding a second adapter
    interpretation. Only one tensor is copied at a time. Training weights,
    optimizer, and RNG are untouched.
    """
    import torch
    from peft.tuners.lora.layer import LoraLayer, ParamWrapper
    from safetensors.torch import save_file

    destination = Path(destination)
    marker = destination / "sdf_export.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["checkpoint"] != checkpoint:
            raise ValueError("Existing merged export belongs to another checkpoint")
        return saved
    if destination.exists():
        raise ValueError(f"Incomplete export requires inspection: {destination}")
    destination.mkdir(parents=True)
    updates = {}
    for layer in model.modules():
        if not isinstance(layer, LoraLayer):
            continue
        if layer.merged:
            raise ValueError("Expected unmerged live training adapters")
        parameter = (
            layer.get_param()
            if isinstance(layer, ParamWrapper)
            else layer.get_base_layer().weight
        )
        if id(parameter) in updates:
            raise ValueError("Multiple updates for one base parameter")
        updates[id(parameter)] = layer
    index, applied, total_bytes = {}, set(), 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".lora_" in name:
                continue
            key = canonical_parameter_name(name)
            if key in index:
                raise ValueError(f"Duplicate exported parameter: {key}")
            tensor = parameter.detach()
            layer = updates.get(id(parameter))
            if layer is not None:
                tensor = tensor + layer.get_delta_weight("default").to(parameter.dtype)
                applied.add(id(parameter))
            tensor = tensor.to("cpu").contiguous()
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite exported weights: {key}")
            shard = f"tensor-{len(index):04d}.safetensors"
            save_file(
                {key: tensor}, str(destination / shard), metadata={"format": "pt"}
            )
            index[key] = shard
            total_bytes += tensor.numel() * tensor.element_size()
            del tensor
    if applied != set(updates):
        raise ValueError("Not every trained adapter update was exported")
    original = json.loads(
        (Path(starting_model) / "model.safetensors.index.json").read_text()
    )
    if set(index) != set(original["weight_map"]):
        raise ValueError("Exported tensors differ from the complete starting model")
    atomic_json(
        destination / "model.safetensors.index.json",
        {"metadata": {"total_size": total_bytes}, "weight_map": index},
    )
    shutil.copyfile(Path(starting_model) / "config.json", destination / "config.json")
    tokenizer.save_pretrained(destination)
    receipt = {
        "checkpoint": checkpoint,
        "starting_model": str(starting_model),
        "applied_updates": len(applied),
        "tensor_count": len(index),
        "tensor_bytes": total_bytes,
        "requantized": False,
        "method": "PEFT get_delta_weight; non-mutating tensor-by-tensor W + delta",
    }
    atomic_json(marker, receipt)
    return receipt
