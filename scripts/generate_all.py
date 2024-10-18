import argparse
import re
from pathlib import Path
from typing import Optional

import numpy
import torch
import yaml
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm

from utility.save_arguments import save_arguments
from yukarin_sosoa.config import Config
from yukarin_sosoa.dataset import (
    F0ProcessMode,
    FeatureTargetDataset,
    SpeakerFeatureDataset,
    create_dataset,
    preprocess,
)
from yukarin_sosoa.generator import Generator, GeneratorOutput


def _extract_number(f):
    s = re.findall(r"\d+", str(f))
    return int(s[-1]) if s else -1


def _get_predictor_model_path(
    model_dir: Path,
    iteration: int = None,
    prefix: str = "predictor_",
):
    if iteration is None:
        paths = model_dir.glob(prefix + "*.pth")
        model_path = list(sorted(paths, key=_extract_number))[-1]
    else:
        model_path = model_dir / (prefix + "{}.pth".format(iteration))
        assert model_path.exists()
    return model_path


def generate_all(
    model_dir: Path,
    model_iteration: Optional[int],
    model_config: Optional[Path],
    dataset_name: str,
    output_dir: Path,
    transpose: bool,
    use_gpu: bool,
):
    if model_config is None:
        model_config = model_dir / "config.yaml"

    output_dir.mkdir(parents=True, exist_ok=True)
    save_arguments(output_dir / "arguments.yaml", generate_all, locals())

    config = Config.from_dict(yaml.safe_load(model_config.open()))

    model_path = _get_predictor_model_path(
        model_dir=model_dir,
        iteration=model_iteration,
    )
    generator = Generator(
        config=config,
        predictor=model_path,
        use_gpu=use_gpu,
    )

    config.dataset.test_num = 0
    config.dataset.valid_num = 9999999
    dataset = create_dataset(config.dataset)[dataset_name]

    if isinstance(dataset, ConcatDataset):
        dataset = dataset.datasets[0]
    if isinstance(dataset.dataset, FeatureTargetDataset):
        inputs = dataset.dataset.datas
        speaker_ids = [None] * len(inputs)
    elif isinstance(dataset.dataset, SpeakerFeatureDataset):
        inputs = dataset.dataset.dataset.datas
        speaker_ids = dataset.dataset.speaker_ids
    else:
        raise ValueError(dataset)

    for input, speaker_id in tqdm(
        zip(inputs, speaker_ids), total=len(inputs), desc="generate_all"
    ):
        input_data = input.generate()
        data = preprocess(
            d=input_data,
            max_sampling_length=99999999,
            prepost_silence_length=99999999,
            f0_process_mode=F0ProcessMode(config.dataset.f0_process_mode),
            time_mask_max_second=0,
            time_mask_rate=0,
        )

        f0 = data["f0"]
        phoneme = data["phoneme"]

        # 長い場合は雑に区切る
        if len(f0) > config.dataset.max_sampling_length:
            num = len(f0) // config.dataset.max_sampling_length
            f0_list = numpy.array_split(f0, num)
            phoneme_list = numpy.array_split(phoneme, num)
        else:
            f0_list = [f0]
            phoneme_list = [phoneme]

        output_list: list[GeneratorOutput] = generator(
            f0_list=f0_list,
            phoneme_list=phoneme_list,
            speaker_id=(
                numpy.array([speaker_id] * len(f0_list))
                if speaker_id is not None
                else None
            ),
        )
        spec = (
            torch.cat([output["spec"] for output in output_list], dim=0)
            .detach()
            .cpu()
            .numpy()
        )

        if transpose:
            spec = spec.T

        name = Path(input.f0_path).stem
        numpy.save(output_dir.joinpath(name + ".npy"), spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("--model_iteration", type=int)
    parser.add_argument("--model_config", type=Path)
    parser.add_argument("--dataset_name", default="train")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--transpose", action="store_true")
    parser.add_argument("--use_gpu", action="store_true")
    generate_all(**vars(parser.parse_args()))
