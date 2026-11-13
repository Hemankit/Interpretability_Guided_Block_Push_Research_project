

from dataclasses import dataclass
import math

from .world import WorldState


@dataclass(frozen=True)
class TaskResult:
    distance_to_goal: float
    inside_goal: bool
    consecutive_inside_checks: int
    success: bool


class BlockPushTask:
    def __init__(
        self,
        goal_x: float,
        goal_y: float,
        goal_radius: float = 0.15,
        required_checks: int = 5,
    ):
        # goal_x and goal_y are retained for compatibility.
        # The actual goal is read from WorldState.

        if not math.isfinite(goal_radius) or goal_radius <= 0:
            raise ValueError(
                "Goal radius must be finite and positive."
            )

        if type(required_checks) is not int or required_checks < 1:
            raise ValueError(
                "Required checks must be a positive integer."
            )

        self.goal_radius = float(goal_radius)
        self.required_checks = required_checks

        self.reset()

    def reset(self):
        self.consecutive_inside_checks = 0
        self.current_distance_to_goal = math.inf
        self.current_inside_goal = False
        self.current_success = False

        self._last_evaluated_step = None
        self._last_result = None

    def evaluate(self, world_state: WorldState) -> TaskResult:

        # Same physics state must not count twice.
        if world_state.step == self._last_evaluated_step:
            return self._last_result

        # Detect accidental episode rollback.
        if (
            self._last_evaluated_step is not None
            and world_state.step < self._last_evaluated_step
        ):
            raise ValueError(
                "Reset the task after resetting the world."
            )

        # Use the goal from the current world state.
        distance_to_goal = math.hypot(
            world_state.block.x - world_state.goal.x,
            world_state.block.y - world_state.goal.y,
        )

        inside_goal = distance_to_goal <= self.goal_radius

        if inside_goal:
            self.consecutive_inside_checks += 1
        else:
            self.consecutive_inside_checks = 0

        success = (
            self.consecutive_inside_checks >= self.required_checks
        )

        self.current_distance_to_goal = distance_to_goal
        self.current_inside_goal = inside_goal
        self.current_success = success
        self._last_evaluated_step = world_state.step

        self._last_result = TaskResult(
            distance_to_goal=distance_to_goal,
            inside_goal=inside_goal,
            consecutive_inside_checks=self.consecutive_inside_checks,
            success=success,
        )

        return self._last_result

    
