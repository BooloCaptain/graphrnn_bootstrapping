import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def binary_cross_entropy_weight(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    has_weight: bool = False,
    weight_length: int = 1,
    weight_max: float = 10.0,
) -> torch.Tensor:
    if has_weight:
        weight = torch.ones_like(y)
        weight_linear = torch.arange(1, weight_length + 1, device=y.device).float() / weight_length * weight_max
        weight_linear = weight_linear.view(1, weight_length, 1).repeat(y.size(0), 1, y.size(2))
        weight[:, -weight_length:, :] = weight_linear
        return F.binary_cross_entropy(y_pred, y, weight=weight)
    return F.binary_cross_entropy(y_pred, y)


def sample_sigmoid(
    y_logits: torch.Tensor,
    sample: bool,
    thresh: float = 0.5,
    sample_time: int = 2,
) -> torch.Tensor:
    y = torch.sigmoid(y_logits)

    if sample:
        if sample_time > 1:
            y_result = torch.rand_like(y)
            for i in range(y_result.size(0)):
                for _ in range(sample_time):
                    y_thresh = torch.rand(y.size(1), y.size(2), device=y.device)
                    y_result[i] = (y[i] > y_thresh).float()
                    if torch.sum(y_result[i]) > 0:
                        break
            return y_result

        y_thresh = torch.rand_like(y)
        return (y > y_thresh).float()

    y_thresh = torch.full_like(y, thresh)
    return (y > y_thresh).float()


class GRUPlain(nn.Module):
    def __init__(
        self,
        input_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        has_input: bool = True,
        has_output: bool = False,
        output_size: int | None = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output
        self.hidden = None

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.GRU(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )

        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, output_size),
            )

        self.relu = nn.ReLU()

        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.25)
            elif "weight" in name:
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain("sigmoid"))

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("relu"))

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def forward(self, input_raw: torch.Tensor, pack: bool = False, input_len: list[int] | None = None) -> torch.Tensor:
        if self.has_input:
            input_tensor = self.relu(self.input(input_raw))
        else:
            input_tensor = input_raw

        if pack:
            input_tensor = pack_padded_sequence(input_tensor, input_len, batch_first=True)

        output_raw, self.hidden = self.rnn(input_tensor, self.hidden)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]

        if self.has_output:
            output_raw = self.output(output_raw)

        return output_raw
