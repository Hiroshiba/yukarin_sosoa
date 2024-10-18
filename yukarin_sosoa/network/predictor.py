import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from ..config import NetworkConfig
from ..network.conformer.encoder import Encoder
from ..network.transformer.utility import make_non_pad_mask


class Postnet(nn.Module):
    def __init__(
        self,
        odim,
        n_layers=5,
        n_chans=512,
        n_filts=5,
        dropout_rate=0.5,
    ):
        super(Postnet, self).__init__()
        postnet = []
        for layer in range(n_layers - 1):
            ichans = odim if layer == 0 else n_chans
            ochans = odim if layer == n_layers - 1 else n_chans
            postnet += [
                nn.Sequential(
                    nn.Conv1d(
                        ichans,
                        ochans,
                        n_filts,
                        stride=1,
                        padding=(n_filts - 1) // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(ochans),
                    nn.Tanh(),
                    nn.Dropout(dropout_rate),
                )
            ]
        ichans = n_chans if n_layers != 1 else odim
        postnet += [
            nn.Sequential(
                nn.Conv1d(
                    ichans,
                    odim,
                    n_filts,
                    stride=1,
                    padding=(n_filts - 1) // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(odim),
                nn.Dropout(dropout_rate),
            )
        ]
        self.postnet = nn.Sequential(*postnet)

    def forward(self, x: Tensor):
        return self.postnet(x)


class Predictor(nn.Module):
    def __init__(
        self,
        input_feature_size: int,
        output_size: int,
        speaker_size: int,
        speaker_embedding_size: int,
        hidden_size: int,
        encoder: Encoder,
    ):
        super().__init__()
        self.output_size = output_size

        self.speaker_embedder = (
            nn.Embedding(
                num_embeddings=speaker_size,
                embedding_dim=speaker_embedding_size,
            )
            if speaker_size > 0
            else None
        )

        input_size = input_feature_size + speaker_embedding_size
        self.pre = nn.Linear(input_size, hidden_size)

        self.encoder = encoder

        self.post = nn.Linear(hidden_size, output_size)

        self.postnet = Postnet(
            odim=output_size,
            n_layers=5,
            n_chans=hidden_size,
            n_filts=5,
            dropout_rate=0.5,
        )

    def forward(
        self,
        f0_list: list[Tensor],  # [(L, )]
        phoneme_list: list[Tensor],  # [(L, )]
        speaker_id: Tensor | None,  # (B, )
    ):
        """
        B: batch size
        L: length
        """
        device = f0_list[0].device

        length = torch.tensor([f0.shape[0] for f0 in f0_list], device=device)

        f0 = pad_sequence(f0_list, batch_first=True)  # (B, L, ?)
        phoneme = pad_sequence(phoneme_list, batch_first=True)  # (B, L, ?)
        h = torch.cat((f0, phoneme), dim=2)  # (B, L, ?)

        if self.speaker_embedder is not None and speaker_id is not None:
            speaker_id = self.speaker_embedder(speaker_id)
            speaker_id = speaker_id.unsqueeze(dim=1)  # (B, 1, ?)
            speaker_feature = speaker_id.expand(
                speaker_id.shape[0], h.shape[1], speaker_id.shape[2]
            )  # (B, L, ?)
            h = torch.cat((h, speaker_feature), dim=2)  # (B, L, ?)

        h = self.pre(h)

        mask = make_non_pad_mask(length).unsqueeze(-2).to(device)  # (B, L, 1)
        h, _ = self.encoder(x=h, cond=None, mask=mask)

        output1 = self.post(h)
        output2 = output1 + self.postnet(output1.transpose(1, 2)).transpose(1, 2)
        return (
            [output1[i, :l] for i, l in enumerate(length)],
            [output2[i, :l] for i, l in enumerate(length)],
        )


def create_predictor(config: NetworkConfig):
    encoder = Encoder(
        hidden_size=config.hidden_size,
        condition_size=0,
        block_num=config.block_num,
        dropout_rate=0.2,
        positional_dropout_rate=0.2,
        attention_head_size=2,
        attention_dropout_rate=0.2,
        use_macaron_style=True,
        use_conv_glu_module=True,
        conv_glu_module_kernel_size=31,
        feed_forward_hidden_size=config.hidden_size * 4,
        feed_forward_kernel_size=3,
    )
    return Predictor(
        input_feature_size=config.input_feature_size,
        output_size=config.output_size,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
        hidden_size=config.hidden_size,
        encoder=encoder,
    )
