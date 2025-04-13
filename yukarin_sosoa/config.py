import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .utility import dataclass_utility
from .utility.git_utility import get_branch_name, get_commit_id


@dataclass
class DatasetConfig:
    root_dir: Path
    f0_pathlist_path: Path
    phoneme_pathlist_path: Path
    spec_pathlist_path: Path
    silence_pathlist_path: Path
    phoneme_list_pathlist_path: Optional[Path]
    volume_pathlist_path: Optional[Path]
    prepost_silence_length: int
    max_sampling_length: int
    f0_process_mode: str
    phoneme_type: str
    time_mask_max_second: float
    time_mask_rate: float
    speaker_dict_path: Optional[Path]
    num_speaker: Optional[int]
    weighted_speaker_id: Optional[int]
    speaker_weight: Optional[int]
    test_num: int
    test_trial_num: int = 1
    valid_f0_pathlist_path: Optional[Path] = None
    valid_phoneme_pathlist_path: Optional[Path] = None
    valid_spec_pathlist_path: Optional[Path] = None
    valid_silence_pathlist_path: Optional[Path] = None
    valid_phoneme_list_pathlist_path: Optional[Path] = None
    valid_volume_pathlist_path: Optional[Path] = None
    valid_speaker_dict_path: Optional[Path] = None
    valid_trial_num: Optional[int] = None
    valid_num: Optional[int] = None
    seed: int = 0


@dataclass
class NetworkConfig:
    input_feature_size: int
    output_size: int
    speaker_size: int
    speaker_embedding_size: int
    hidden_size: int
    block_num: int
    dropout_rate: float


@dataclass
class ModelConfig:
    pass


@dataclass
class TrainConfig:
    batch_size: int
    eval_batch_size: int
    log_epoch: int
    eval_epoch: int
    snapshot_epoch: int
    stop_epoch: int
    model_save_num: int
    optimizer: Dict[str, Any]
    scheduler: Dict[str, Any]
    weight_initializer: Optional[str] = None
    pretrained_predictor_path: Optional[Path] = None
    num_processes: int = 4
    use_gpu: bool = True
    use_amp: bool = True


@dataclass
class ProjectConfig:
    name: str
    tags: Dict[str, Any] = field(default_factory=dict)
    category: Optional[str] = None


@dataclass
class Config:
    dataset: DatasetConfig
    network: NetworkConfig
    model: ModelConfig
    train: TrainConfig
    project: ProjectConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        backward_compatible(d)
        return dataclass_utility.convert_from_dict(cls, copy.deepcopy(d))

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_utility.convert_to_dict(self)

    def add_git_info(self):
        self.project.tags["git-commit-id"] = get_commit_id()
        self.project.tags["git-branch-name"] = get_branch_name()


def backward_compatible(d: Dict[str, Any]):
    if "prepost_silence_length" not in d["dataset"]:
        d["dataset"]["prepost_silence_length"] = 99999999

    if "max_sampling_length" not in d["dataset"]:
        d["dataset"]["max_sampling_length"] = 99999999

    if "eval_batch_size" not in d["train"]:
        d["train"]["eval_batch_size"] = d["train"]["batch_size"]
    if "log_iteration" in d["train"]:
        d["train"]["log_epoch"] = d["train"].pop("log_iteration")
    if "eval_iteration" in d["train"]:
        d["train"]["eval_epoch"] = d["train"].pop("eval_iteration")
    if "snapshot_iteration" in d["train"]:
        d["train"]["snapshot_epoch"] = d["train"].pop("snapshot_iteration")
    if "stop_iteration" in d["train"]:
        d["train"]["stop_epoch"] = d["train"].pop("stop_iteration")
    if "model_save_num" not in d["train"]:
        d["train"]["model_save_num"] = 99999999
    if "scheduler" not in d["train"]:
        d["train"]["scheduler"] = {}
    if "step_shift" in d["train"]:
        del d["train"]["step_shift"]
    if "noam_shift" in d["train"]:
        del d["train"]["noam_shift"]
    if "use_multithread" in d["train"]:
        del d["train"]["use_multithread"]
    if "optuna" in d["train"]:
        del d["train"]["optuna"]

    if "dropout_rate" not in d["network"]:
        d["network"]["dropout_rate"] = 0.2

    if "root_dir" not in d["dataset"]:
        d["dataset"]["root_dir"] = Path(".")
    for before_key in [
        "f0_glob",
        "phoneme_glob",
        "spec_glob",
        "silence_glob",
        "phoneme_list_glob",
        "volume_glob",
        "valid_f0_glob",
        "valid_phoneme_glob",
        "valid_spec_glob",
        "valid_silence_glob",
        "valid_phoneme_list_glob",
        "valid_volume_glob",
    ]:
        if before_key in d["dataset"]:
            after_key = before_key.replace("_glob", "_pathlist_path")
            d["dataset"][after_key] = d["dataset"].pop(before_key)
