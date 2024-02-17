from dataclasses import dataclass


@dataclass
class MiniGPTConfig:
    sequence_length: int = 1024
    vocab_size: int = 65
    embedding_dimension: int = 768
    n_layers: int = 12
    n_heads: int = 12
    dropout: float = 0.0
    bias: bool = True
