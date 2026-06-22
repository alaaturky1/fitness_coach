from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EmaSmoother:
    """
    Exponential moving average smoother for named scalar signals.

    alpha in (0,1]: higher alpha -> less smoothing, lower latency.
    """

    alpha: float = 0.35
    values: dict[str, float] = field(default_factory=dict)

    def update(self, key: str, x: float) -> float:
        prev = self.values.get(key)
        if prev is None:
            self.values[key] = x
            return x
        y = self.alpha * x + (1.0 - self.alpha) * prev
        self.values[key] = y
        return y

    def update_many(self, xs: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in xs.items():
            out[k] = self.update(k, float(v))
        return out

