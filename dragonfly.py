"""Dragonfly agent (candidate solution) for the Dragonfly Algorithm."""

import numpy as np


class Dragonfly:
    """A single dragonfly in the swarm.

    The position vector encodes one candidate routing decision: a continuous
    index into the (dynamic) server pool, mapped to a discrete slot at
    evaluation time.
    """

    def __init__(self, bounds: tuple, dimensions: int = 1) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be >= 1")
        lower: float = float(bounds[0])
        upper: float = float(bounds[1])
        if lower > upper:
            raise ValueError("bounds must be (lower, upper) with lower <= upper")
        # Random initial position within the bounds, per dimension.
        self.position: np.ndarray = np.random.uniform(lower, upper, size=dimensions)
        # Step vector (velocity analogue), initialized in [-1.0, 1.0].
        self.step: np.ndarray = np.random.uniform(-1.0, 1.0, size=dimensions)
        # Fitness is maximized; -inf forces the first evaluation to win.
        self.fitness: float = float("-inf")
