"""Token estimation + configurable LLM cost model.

Token counts here are *estimates* based on character counts. For Persian
text on GPT-4o-family tokenizers, ~3.2 characters per token is a realistic
operating point; tune CHARS_PER_TOKEN for your model. Real API `usage`
numbers always win over these estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CHARS_PER_TOKEN = 3.2

USD_TO_IRR = 230_000  # 1 USD in Toman (project-configured rate)


@dataclass
class LLMCost:
    """Pricing configuration: USD per 1M tokens."""

    input_price_per_1m: float = 2.50
    output_price_per_1m: float = 10.00
    usd_to_irr: int = USD_TO_IRR

    def input_cost_usd(self, tokens: int) -> float:
        return tokens / 1_000_000 * self.input_price_per_1m

    def output_cost_usd(self, tokens: int) -> float:
        return tokens / 1_000_000 * self.output_price_per_1m

    def toman(self, usd: float) -> int:
        return round(usd * self.usd_to_irr)


def estimate_tokens(text: str, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """Estimate token count from character length."""
    if not text:
        return 0
    return max(1, round(len(text) / chars_per_token))


def estimate_tokens_many(texts: list[str], chars_per_token: float = CHARS_PER_TOKEN) -> int:
    return sum(estimate_tokens(t, chars_per_token) for t in texts)


@dataclass
class ScenarioUsage:
    """Token/cost accounting for one benchmark scenario."""

    name: str
    context_tokens: int
    question_tokens: int
    answer_tokens: int
    cost: LLMCost = field(default_factory=LLMCost)

    @property
    def input_tokens(self) -> int:
        return self.context_tokens + self.question_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.answer_tokens

    @property
    def input_cost_usd(self) -> float:
        return self.cost.input_cost_usd(self.input_tokens)

    @property
    def output_cost_usd(self) -> float:
        return self.cost.output_cost_usd(self.answer_tokens)

    @property
    def total_cost_usd(self) -> float:
        return self.input_cost_usd + self.output_cost_usd

    @property
    def total_cost_toman(self) -> int:
        return self.cost.toman(self.total_cost_usd)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "context_tokens": self.context_tokens,
            "question_tokens": self.question_tokens,
            "answer_tokens": self.answer_tokens,
            "input_tokens": self.input_tokens,
            "total_tokens": self.total_tokens,
            "input_cost_usd": round(self.input_cost_usd, 4),
            "output_cost_usd": round(self.output_cost_usd, 4),
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_cost_toman": self.total_cost_toman,
        }
