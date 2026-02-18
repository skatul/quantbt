from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterSpace:
    """Defines a named parameter with a list of candidate values."""

    name: str
    values: list

    @staticmethod
    def range(name: str, start: int | float, stop: int | float,
              step: int | float = 1) -> ParameterSpace:
        """Create parameter space from a numeric range (inclusive of start, exclusive of stop)."""
        vals: list = []
        v = start
        while v < stop:
            vals.append(v)
            v += step
            # Round to avoid floating-point drift
            v = round(v, 10)
        return ParameterSpace(name=name, values=vals)

    @staticmethod
    def choices(name: str, options: list) -> ParameterSpace:
        """Create parameter space from a list of discrete choices."""
        return ParameterSpace(name=name, values=list(options))
