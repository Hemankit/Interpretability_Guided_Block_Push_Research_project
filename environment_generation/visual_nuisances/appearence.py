from dataclasses import dataclass
from logging import config
import random
from sim.world import World
import pybullet as p
import numpy as np
@dataclass(frozen=True)
class Appearance:
    block_color: tuple[int, int, int]
    floor_color: tuple[int, int, int]


def sample_appearance(
    config,
    rng: np.random.Generator,
) -> Appearance:

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    def validate_palette(palette, name):
        if not isinstance(palette, (list, tuple)) or not palette:
            raise ValueError(f"{name} must be a non-empty palette")

        for color in palette:
            if not isinstance(color, (list, tuple)) or len(color) != 3:
                raise ValueError(f"Invalid RGB color in {name}: {color}")

            if any(type(channel) is not int or not 0 <= channel <= 255
                   for channel in color):
                raise ValueError(f"Invalid RGB values in {name}: {color}")

    block_palette = config["block_colors"]
    floor_palette = config["floor_colors"]

    validate_palette(block_palette, "block_colors")
    validate_palette(floor_palette, "floor_colors")

    block_color = tuple(
        block_palette[int(rng.integers(0, len(block_palette)))]
    )

    floor_color = tuple(
        floor_palette[int(rng.integers(0, len(floor_palette)))]
    )

    return Appearance(
        block_color=block_color,
        floor_color=floor_color,
    )

def apply_appearance(world: World, appearance: Appearance) -> None:

    def to_rgba(rgb: tuple[int, int, int]) -> tuple[float, ...]:
        return tuple(channel / 255.0 for channel in rgb) + (1.0,)

    # Change block color
    p.changeVisualShape(
        world.block_id,
        -1,
        rgbaColor=to_rgba(appearance.block_color),
        physicsClientId=world.physics_client,
    )

    # Change floor color
    p.changeVisualShape(
        world.plane_id,
        -1,
        rgbaColor=to_rgba(appearance.floor_color),
        physicsClientId=world.physics_client,
    )
    