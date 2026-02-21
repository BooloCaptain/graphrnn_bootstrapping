from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    graph_type: str = "barabasi_small"
    hidden_size_rnn: int = 128
    hidden_size_rnn_output: int = 16
    embedding_size_rnn: int = 64
    embedding_size_rnn_output: int = 8
    embedding_size_output: int = 64
    num_layers: int = 4

    batch_size: int = 32
    test_batch_size: int = 32
    test_total_size: int = 256
    num_workers: int = 0
    batch_ratio: int = 32
    epochs: int = 3000
    epochs_test_start: int = 100
    epochs_test: int = 100
    epochs_save: int = 100
    epochs_log: int = 10

    lr: float = 0.003
    milestones: list[int] = field(default_factory=lambda: [400, 1000])
    lr_rate: float = 0.3

    max_num_node: int | None = None
    max_prev_node: int = 40

    seed: int = 123
    cuda: int = 0

    output_dir: Path = Path("./")
    save_checkpoints: bool = True

    @property
    def model_save_path(self) -> Path:
        return self.output_dir / "model_save"

    @property
    def graph_save_path(self) -> Path:
        return self.output_dir / "graphs"

    @property
    def fname(self) -> str:
        return f"GraphRNN_RNN_{self.graph_type}_{self.num_layers}_{self.hidden_size_rnn}_"

    @property
    def fname_pred(self) -> str:
        return f"GraphRNN_RNN_{self.graph_type}_{self.num_layers}_{self.hidden_size_rnn}_pred_"
