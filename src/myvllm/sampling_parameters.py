from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64  # Maximum number of completion tokens to generate.
    ignore_eos: bool = False
    max_model_length: int | None = None  # Prompt plus completion length limit.

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError(
                "temperature must be non-negative; 0 selects greedy argmax"
            )
        if self.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if self.max_model_length is not None and self.max_model_length <= 0:
            raise ValueError("max_model_length must be positive when provided")
