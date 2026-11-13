import numpy as np
from dataclasses import dataclass

from sim.task import TaskResult
from sim.world import World, Vec2
from sim.camera import PusherCamera
from sim.task import BlockPushTask
from environment_generation.scenario_generation import Scenario, apply_scenario

@dataclass(frozen=True)
class StepResult:
    observation: np.ndarray
    task_result: TaskResult
    terminated: bool
    truncated: bool


class BlockPushEnv:
    def __init__(
        self,
        pusher_position: Vec2,
        block_position: Vec2,
        goal_position: Vec2,
        gui: bool = False,
        max_actions: int = 200,
        scenario: Scenario | None = None,
    ):
        if isinstance(max_actions, bool) or not isinstance(max_actions, int):
            raise TypeError("max_actions must be an integer")
        if max_actions <= 0:
            raise ValueError("max_actions must be positive")

        self.world = World(
            pusher_position=pusher_position,
            block_position=block_position,
            goal_position=goal_position,
            gui=gui,
        )
        self.camera = PusherCamera(
            self.world.pusher,
            self.world.physics_client,
        )
        self.task = BlockPushTask(
            goal_x=goal_position.x,
            goal_y=goal_position.y,
            goal_radius=0.15,
            required_checks=5,
        )

        self.max_actions = max_actions
        self.action_count = 0
        self.terminated = False
        self.truncated = False
        self._closed = False
        self.scenario = None
        if scenario is not None:
            self.set_scenario(scenario)

    def set_scenario(self, scenario: Scenario | None) -> None:
        """Switch nuisance configuration without leaking previous state.

        Moving from one enabled configuration to a different one is refused;
        the caller must first set None (or an all-disabled scenario), which
        restores the original visuals, and then enable the new configuration.
        """
        if self._closed:
            raise RuntimeError("Environment is closed.")
        current_active = self.scenario is not None and self.scenario.any_enabled
        new_active = scenario is not None and scenario.any_enabled
        if current_active and new_active and scenario != self.scenario:
            raise RuntimeError(
                "Nuisances are already enabled. Disable them (set_scenario(None)) "
                "to restore the default environment before enabling a new "
                "configuration."
            )
        self.scenario = scenario
        apply_scenario(self.world, scenario)

    def reset(self) -> np.ndarray:
        if self._closed:
            raise RuntimeError("Environment is closed.")

        self.world.reset()
        apply_scenario(self.world, self.scenario)
        self.task.reset()
        self.action_count = 0
        self.terminated = False
        self.truncated = False
        return self._capture()

    def step(self, action: tuple[float, float]) -> StepResult:
        if self._closed:
            raise RuntimeError("Environment is closed.")
        if self.terminated or self.truncated:
            raise RuntimeError("Episode finished. Call reset first.")

        vx, vy = action
        if not np.isfinite((vx, vy)).all():
            raise ValueError("Velocity must be finite")

        world_state = self.world.step_action(vx, vy)
        self.action_count += 1
        task_result = self.task.evaluate(world_state)

        self.terminated = task_result.success
        self.truncated = (
            not self.terminated and self.action_count >= self.max_actions
        )

        return StepResult(
            observation=self._capture(),
            task_result=task_result,
            terminated=self.terminated,
            truncated=self.truncated,
        )

    def _capture(self) -> np.ndarray:
        lighting = (
            self.scenario.lighting
            if self.scenario is not None and self.scenario.lighting_enabled
            else None
        )
        return self.camera.capture(lighting=lighting)

    def close(self) -> None:
        if not self._closed:
            self.world.close()
            self._closed = True

if __name__ == "__main__":
    env = BlockPushEnv(
        pusher_position=Vec2(0, 0),
        block_position=Vec2(1, 1),
        goal_position=Vec2(2, 2),
        gui=True,
    )
    try:
        obs = env.reset()
        print("obs shape:", obs.shape)
        result = None
        while not (env.terminated or env.truncated):
            result = env.step((0.5, 0.5))  # naive diagonal push
        print("steps:", env.action_count,
              "terminated:", env.terminated, "truncated:", env.truncated)
        print(result.task_result)
    finally:
        env.close()

