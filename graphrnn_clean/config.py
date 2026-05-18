from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    graph_type: str = "grid"  # Set default to grid
    hidden_size_rnn: int = 128  # Matches paper (Larger Model)
    hidden_size_rnn_output: int = 16  # Matches paper
    embedding_size_rnn: int = 64
    embedding_size_rnn_output: int = 8  # Matches paper's 8D projection
    embedding_size_output: int = 64
    num_layers: int = 4  # Matches paper

    batch_size: int = 32  # Our anti-clone hardware optimization
    test_batch_size: int = 32
    test_total_size: int = 1000
    num_workers: int = 4
    batch_ratio: int = 32  # 80 * 13 = 1040 graphs per epoch (13 steps/epoch)
    epochs: int = 3000  # Matches the total 3 million graph volume of the paper
    epochs_test_start: int = 100
    epochs_test: int = 100
    epochs_save: int = 100
    epochs_log: int = 100

    lr: float = 0.009  # Exact paper starting learning rate
    # Paper decayed at 13.3% and 33.3% of total training volume
    milestones: list[int] = field(default_factory=lambda: [400, 1000]) 
    lr_rate: float = 0.3  # Exact paper decay multiplier
    amp_mode: str = "off"  # Force FP32 to prevent grid counting amnesia
    grad_set_to_none: bool = True

    max_num_node: int | None = None
    max_prev_node: int = 40  # Exact paper cutoff limit for Grids (M=40)

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