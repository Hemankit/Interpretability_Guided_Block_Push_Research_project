"""Run short pusher episodes, save camera frames, and check nuisance state leakage.

Sequence: baseline -> nuisances A -> (switching straight to B must be refused)
-> disabled (must equal baseline) -> nuisances B -> disabled (must equal baseline).
Task completion is irrelevant; the pusher just drives forward.

Usage: python scripts/capture_nuisance_episode.py [--steps 40] [--every 10]
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment_generation.scenario_generation import (  # noqa: E402
    load_scenario_config,
    sample_scenario,
)
from sim.block_push_env import BlockPushEnv  # noqa: E402
from sim.world import Vec2  # noqa: E402

CONFIG_PATH = ROOT / "environment_generation" / "nuisance_classes.yaml"


def enabled_config(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    for nuisance in cfg["nuisances"].values():
        nuisance["enabled"] = True
    return cfg


def save(frame: np.ndarray, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(folder / f"{name}.png")


def run_episode(env, label: str, out_dir: Path, steps: int, every: int):
    """Reset, drive +x for `steps` actions, saving frames. Returns first frame."""
    folder = out_dir / label
    first = env.reset()
    save(first, folder, "step_000")
    for i in range(1, steps + 1):
        result = env.step((0.5, 0.0))
        if i % every == 0 or i == steps:
            save(result.observation, folder, f"step_{i:03d}")
        if result.terminated or result.truncated:
            break
    return first


def world_is_default(env) -> bool:
    world = env.world
    if world.distractor_ids:
        return False
    for body_id, rgba in world._default_rgba.items():
        current = tuple(
            p.getVisualShapeData(body_id, physicsClientId=world.physics_client)[0][7]
        )
        if not np.allclose(current, rgba):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--out", type=Path, default=ROOT / "images")
    parser.add_argument("--seed-a", type=int, default=1)
    parser.add_argument("--seed-b", type=int, default=2)
    args = parser.parse_args()

    config = enabled_config(load_scenario_config(CONFIG_PATH))
    scenario_a = sample_scenario(config, args.seed_a)
    scenario_b = sample_scenario(config, args.seed_b)
    if scenario_a == scenario_b:
        print("seed-a and seed-b produced identical scenarios; pick other seeds")
        return 2

    env = BlockPushEnv(
        pusher_position=Vec2(0.5, 1.0),
        block_position=Vec2(1.0, 1.0),
        goal_position=Vec2(1.8, 1.0),
        gui=False,
        max_actions=args.steps + 1,
    )
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        print(("PASS  " if condition else "FAIL  ") + message)
        if not condition:
            failures.append(message)

    def episode(label: str) -> np.ndarray:
        return run_episode(env, label, args.out, args.steps, args.every)

    try:
        baseline = episode("1_baseline")
        check(world_is_default(env), "baseline world is in default state")

        env.set_scenario(scenario_a)
        frame_a = episode("2_nuisances_A")
        check(not np.array_equal(frame_a, baseline), "nuisances A change the camera image")
        check(not world_is_default(env), "nuisances A modify the world")

        try:
            env.set_scenario(scenario_b)
            check(False, "switching A -> B directly is refused")
        except RuntimeError:
            check(True, "switching A -> B directly is refused")
        frame_still_a = episode("3_refused_switch_still_A")
        check(np.array_equal(frame_still_a, frame_a), "refused switch leaves nuisances A unchanged")

        env.set_scenario(None)
        check(world_is_default(env), "disabling restores default world state immediately")
        frame_off = episode("4_disabled_after_A")
        check(np.array_equal(frame_off, baseline), "disabled after A matches baseline image exactly")

        env.set_scenario(scenario_b)
        frame_b = episode("5_nuisances_B")
        check(not np.array_equal(frame_b, baseline), "nuisances B change the camera image")
        check(not np.array_equal(frame_b, frame_a), "nuisances B differ from A (no stale A state)")

        env.set_scenario(None)
        frame_off2 = episode("6_disabled_after_B")
        check(np.array_equal(frame_off2, baseline), "disabled after B matches baseline image exactly")
        check(world_is_default(env), "world back to default state at the end")
    finally:
        env.close()

    print(f"\nImages written to {args.out}")
    print("RESULT:", "FAILED" if failures else "ALL CHECKS PASSED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
