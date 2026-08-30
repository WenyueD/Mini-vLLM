from dataclasses import dataclass


@dataclass(frozen=True)
class SpeculativeConfig:
    enabled: bool = False
    proposer: str = "ngram"
    num_speculative_tokens: int = 4
    ngram_min_size: int = 2
    ngram_max_size: int = 4

    def __post_init__(self) -> None:
        if self.proposer not in {"ngram", "mock"}:
            raise ValueError(f"unsupported proposer: {self.proposer}")
        if self.num_speculative_tokens < 0:
            raise ValueError("num_speculative_tokens must be non-negative")
        if self.ngram_min_size <= 0:
            raise ValueError("ngram_min_size must be positive")
        if self.ngram_max_size < self.ngram_min_size:
            raise ValueError("ngram_max_size must be >= ngram_min_size")

    @classmethod
    def from_value(
        cls, value: "SpeculativeConfig | dict | None"
    ) -> "SpeculativeConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError("speculative_config must be a dict or SpeculativeConfig")
