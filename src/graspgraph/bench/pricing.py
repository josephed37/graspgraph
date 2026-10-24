"""Token prices, read from prices.toml next to this module."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

PRICES_FILE = Path(__file__).with_name("prices.toml")


@dataclass(frozen=True)
class Price:
    input: float  # USD per million input tokens
    output: float  # USD per million output tokens
    max_prompt_tokens: int | None = None  # the rate only holds up to this prompt size

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """USD cost of one call."""
        return (input_tokens * self.input + output_tokens * self.output) / 1_000_000

    def note(self, input_tokens: int) -> str | None:
        """A warning when the prompt is too long for this rate to be trusted."""
        if self.max_prompt_tokens is not None and input_tokens > self.max_prompt_tokens:
            return f"prompt of {input_tokens} tokens exceeds the {self.max_prompt_tokens}-token price tier; cost may be too low"
        return None


PriceTable = dict[str, dict[str, Price]]


def load_prices(path: Path = PRICES_FILE) -> PriceTable:
    """Parse a prices TOML file into backend -> model -> Price."""
    raw = tomllib.loads(path.read_text())
    return {backend: {model: Price(**fields) for model, fields in models.items()} for backend, models in raw.items()}


def price_for(prices: PriceTable, backend: str, model: str) -> Price:
    """The price of a model; raises ValueError for an unpriced model, since the spend cap needs one."""
    models = prices.get(backend, {})
    if model in models:
        return models[model]
    if "*" in models:
        return models["*"]
    raise ValueError(f"no price for {backend}/{model}; add it to {PRICES_FILE.name}")
