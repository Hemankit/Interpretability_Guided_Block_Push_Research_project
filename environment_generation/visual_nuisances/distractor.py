from dataclasses import dataclass
import numpy as np
import pybullet as p
from sim.world import World
@dataclass(frozen=True)
class Distractor:
    position: tuple[float, float]
    color: tuple[int, int, int]
    size: float


def validate_distractor_properties(config: dict) -> None:

    # 1. Validate configuration
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")

    # 2. Retrieve parameters
    counts = config["counts"]
    positions = config["candidate_positions"]
    colors = config["colors"]
    sizes = config["sizes"]


    # 3. Reusable numeric validation
    def valid_number(value):
        return (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value)
        )

    # Validate positions (may be empty when only zero distractors are possible)
    if not isinstance(positions, (list, tuple)):
        raise ValueError("positions must be a list")

    for position in positions:
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            raise ValueError("Each position must contain two coordinates")
        if not all(valid_number(v) for v in position):
            raise ValueError("Position coordinates must be finite numbers")

    if len(set(map(tuple, positions))) != len(positions):
        raise ValueError("Duplicate positions are not allowed")

    # Validate counts: nonnegative integers (0 means no distractors)
    if not isinstance(counts, (list, tuple)) or not counts:
        raise ValueError("counts must be a non-empty list")

    for count in counts:
        if (
            not isinstance(count, (int, np.integer))
            or isinstance(count, (bool, np.bool_))
            or count < 0
        ):
            raise ValueError("Each count must be a nonnegative integer")
        if count > len(positions):
            raise ValueError("Each count must not exceed the number of available positions")

    # Colors and sizes are only needed when at least one distractor can be drawn
    needs_appearance = any(count > 0 for count in counts)

    if not isinstance(colors, (list, tuple)) or (needs_appearance and not colors):
        raise ValueError("colors must be a non-empty list")

    for color in colors:
        if not isinstance(color, (list, tuple)) or len(color) != 3:
            raise ValueError("Each color must contain three RGB channels")
        if any(
            type(channel) is not int or not 0 <= channel <= 255
            for channel in color
        ):
            raise ValueError("RGB channels must be integers in [0, 255]")
    # sizes represents the full side length, in meters, of a square visual distractor
    if not isinstance(sizes, (list, tuple)) or (needs_appearance and not sizes):
        raise ValueError("sizes must be a non-empty list")

    for size in sizes:
        if not valid_number(size) or size <= 0:
            raise ValueError("Distractor sizes must be finite and positive")

def sample_distractors(
    config: dict,
    rng: np.random.Generator,
) -> tuple[Distractor, ...]:
    if not isinstance(rng, np.random.Generator):
      raise TypeError("rng must be a numpy.random.Generator")
    validate_distractor_properties(config)
    counts = config["counts"]
    positions = config["candidate_positions"]
    colors = config["colors"]
    sizes = config["sizes"]

    count = int(counts[int(rng.integers(0, len(counts)))])
    if count == 0:
        return ()
    indices = rng.choice(len(positions), size=count, replace=False)

    distractors = []

    for i in indices:
      position = tuple(float(v) for v in positions[int(i)])
      color = tuple(
        int(v) for v in colors[int(rng.integers(0, len(colors)))]
    )
      size = float(sizes[int(rng.integers(0, len(sizes)))])

      distractors.append(Distractor(position, color, size))

    return tuple(distractors)

def apply_distractors(
    world: World,
    distractors: tuple[Distractor, ...],
) -> None:
    # Remove previous distractors
    for body_id in getattr(world, "distractor_ids", []):
        p.removeBody(
            body_id,
            physicsClientId=world.physics_client,
        )

    world.distractor_ids = []

    thickness = 0.005
    clearance = 0.001

    for distractor in distractors:
        x, y = distractor.position

        visual_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[
                distractor.size / 2,
                distractor.size / 2,
                thickness / 2,
            ],
            rgbaColor=[
                c / 255.0 for c in distractor.color
            ] + [1.0],
            physicsClientId=world.physics_client,
        )

        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_id,
            basePosition=[
                x,
                y,
                thickness / 2 + clearance,
            ],
            physicsClientId=world.physics_client,
        )

        world.distractor_ids.append(body_id)
    

    
