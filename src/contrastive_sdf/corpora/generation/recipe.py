"""Generation recipes contain authoring controls, never training/eval controls."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import yaml
from pydantic import Field

from contrastive_sdf.config.schema import DatasetSpec
from contrastive_sdf.config.types import PreferenceMapping, StrictModel
from contrastive_sdf.corpora.generation.schema import AtomicPipelineConfig


class GenerationCorpus(StrictModel):
    version: str
    directory: str
    document_count: int = Field(gt=0)
    tokenizer: str
    sha256: dict[str, str | None]
    atomic: AtomicPipelineConfig


class GenerationRecipe(StrictModel):
    recipe_version: int = Field(ge=1, le=1)
    mode: str
    target_base_model: str
    universes: dict[str, PreferenceMapping]
    corpus: GenerationCorpus
    overlap_dataset: DatasetSpec

    @property
    def models(self):
        return [SimpleNamespace(base_model=self.target_base_model)]

    @property
    def evaluation(self):
        return SimpleNamespace(dataset=self.overlap_dataset)


class ExperimentPlan:
    def __init__(self, source, recipe):
        self.source = str(source)
        self.contract = recipe
        self.contract_sha256 = hashlib.sha256(Path(source).read_bytes()).hexdigest()

    def require_ready_for_training(self):
        if not all(self.contract.corpus.sha256.values()):
            raise ValueError(
                "generation corpus hashes are unset; export then freeze an execution config"
            )


def load_experiment_plan(path):
    path = Path(path)
    raw = yaml.safe_load(path.read_bytes())
    if raw.get("contract_version") == 2:
        # Immutable imported source config, used only to recover upstream provenance.
        raw = {
            "recipe_version": 1,
            "mode": raw["mode"],
            "target_base_model": raw["models"][0]["base_model"],
            "universes": raw["universes"],
            "corpus": {
                k: raw["corpus"][k]
                for k in [
                    "version",
                    "directory",
                    "document_count",
                    "tokenizer",
                    "sha256",
                    "atomic",
                ]
            },
            "overlap_dataset": raw["evaluation"]["dataset"],
        }
    return ExperimentPlan(path, GenerationRecipe.model_validate(raw))
