from dataclasses import dataclass


@dataclass
class WarpPipeConfig:
    enabled: bool = True
    rank: int = 32
    num_staging_slots: int = 8
    min_segment_tokens: int = 4
