from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True)
class Lighting:
    direction: tuple[float, float, float]
    ambient: float
    diffuse: float


from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Lighting:
    direction: tuple[float, float, float]
    ambient: float
    diffuse: float


# PyBullet TinyRenderer's built-in light for a z-up world (verified pixel-exact).
DEFAULT_LIGHTING = Lighting(direction=(-50.0, 30.0, 100.0), ambient=0.6, diffuse=0.35)


def sample_lighting(
    config,
    rng: np.random.Generator,
) -> Lighting:

    # 1. Validate RNG
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of np.random.Generator")

    # 2. Retrieve lighting parameters
    directions = config["directions"]
    ambient_coefficients = config["ambient_coefficients"]
    diffuse_coefficients = config["diffuse_coefficients"]

    # 3. Validate parameters
    def valid_number(value):
        return (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value)
        )

    def validate_directions(directions):
        if not isinstance(directions, (list, tuple)) or not directions:
            raise ValueError("directions must be a non-empty list")

        for direction in directions:
            if not isinstance(direction, (list, tuple)) or len(direction) != 3:
                raise ValueError("Each direction must have three components")

            if not all(valid_number(v) for v in direction):
                raise ValueError("Direction components must be finite numbers")

            if all(v == 0 for v in direction):
                raise ValueError("Direction cannot be the zero vector")

    def validate_coefficients(values, name):
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"{name} must be a non-empty list")

        for value in values:
            if not valid_number(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must contain finite numbers between 0 and 1"
                )

    validate_directions(directions)
    validate_coefficients(ambient_coefficients, "ambient_coefficients")
    validate_coefficients(diffuse_coefficients, "diffuse_coefficients")

    # Sample direction, rejecting zero vectors
    index = int(rng.integers(0, len(directions)))
    direction = tuple(float(v) for v in directions[index])

# Sample ambient coefficient
    index = int(rng.integers(0, len(ambient_coefficients)))
    ambient = float(ambient_coefficients[index])

# Sample diffuse coefficient
    index = int(rng.integers(0, len(diffuse_coefficients)))
    diffuse = float(diffuse_coefficients[index])

    return Lighting(
        direction=direction,
        ambient=ambient,
        diffuse=diffuse,
    )

