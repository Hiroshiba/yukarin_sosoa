from typing import TypedDict

import torch
from torch import Tensor, nn

from yukarin_sosoa.dataset import DatasetOutput

from .generator import Generator, GeneratorOutput


class EvaluatorOutput(TypedDict):
    value: Tensor
    data_num: int


class Evaluator(nn.Module):
    def __init__(self, generator: Generator):
        super().__init__()
        self.generator = generator

    def forward(self, data: DatasetOutput) -> EvaluatorOutput:
        output_list: list[GeneratorOutput] = self.generator(
            f0_list=data["f0"],
            phoneme_list=data["phoneme"],
            speaker_id=(
                torch.stack(data["speaker_id"])
                if data["speaker_id"] is not None
                else None
            ),
        )

        output = torch.cat([output["spec"] for output in output_list])

        target_spec = torch.cat(data["spec"])

        value = torch.abs(output - target_spec).mean()

        return EvaluatorOutput(
            value=value,
            data_num=len(data),
        )
