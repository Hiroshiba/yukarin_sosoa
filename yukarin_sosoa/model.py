from typing import Any, TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from yukarin_sosoa.dataset import DatasetOutput

from .config import ModelConfig
from .network.predictor import Predictor


class ModelOutput(TypedDict):
    loss: Tensor
    loss1: Tensor
    loss2: Tensor
    data_num: int


def reduce_result(results: list[ModelOutput]):
    result: dict[str, Any] = {}
    sum_data_num = sum([r["data_num"] for r in results])
    for key in set(results[0].keys()) - {"data_num"}:
        values = [r[key] * r["data_num"] for r in results]
        if isinstance(values[0], Tensor):
            result[key] = torch.stack(values).sum() / sum_data_num
        else:
            result[key] = sum(values) / sum_data_num
    return result


class Model(nn.Module):
    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, data: DatasetOutput) -> ModelOutput:
        batch_size = len(data["spec"])

        output1_list, output2_list = self.predictor(
            f0_list=data["f0"],
            phoneme_list=data["phoneme"],
            speaker_id=(
                torch.stack(data["speaker_id"])
                if data["speaker_id"] is not None
                else None
            ),
        )

        output1 = torch.cat(output1_list)
        output2 = torch.cat(output2_list)

        target_spec = torch.cat(data["spec"])

        loss1 = F.l1_loss(input=output1, target=target_spec)
        loss2 = F.l1_loss(input=output2, target=target_spec)
        loss = loss1 + loss2

        return ModelOutput(
            loss=loss,
            loss1=loss1,
            loss2=loss2,
            data_num=batch_size,
        )
