
from hashlib import sha256
import numpy as np


import secrets


class SeedManager:
    def __init__(self, root_seed: int | None):
        if root_seed is not None:
            if isinstance(root_seed, (bool, np.bool_)) or not isinstance(
                root_seed, (int, np.integer)
            ):
                raise TypeError("root_seed must be an integer or None")

            if root_seed < 0:
                raise ValueError("root_seed must be non-negative")

        self.root_seed = (
            int(root_seed)
            if root_seed is not None
            else secrets.randbits(32)
        )

    def rng(self, name: str) -> np.random.Generator:
        if not isinstance(name, str):
            raise TypeError("name must be a string")

        if not name.strip():
            raise ValueError("name must not be blank")

        seed = int(
            sha256(f"{self.root_seed}_{name}".encode()).hexdigest(),
            16
        )

        return np.random.default_rng(seed)