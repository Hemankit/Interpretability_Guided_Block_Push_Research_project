import math

import pytest

from sim.block_push_env import BlockPushEnv
from sim.world import Vec2, World


@pytest.fixture
def world():
    w = World(
        pusher_position=Vec2(0.0, 0.0),
        block_position=Vec2(2.0, 2.0),
        goal_position=Vec2(3.0, 3.0),
        gui=False,
    )
    yield w
    w.close()


@pytest.fixture
def env():
    # Pusher, block and goal are collinear along +x so a straight push works.
    e = BlockPushEnv(
        pusher_position=Vec2(0.5, 1.0),
        block_position=Vec2(1.0, 1.0),
        goal_position=Vec2(1.8, 1.0),
        gui=False,
        max_actions=300,
    )
    e.reset()
    yield e
    e.close()


def test_pusher_stays_put_with_zero_action(world):
    start = world.get_world_state().pusher
    for _ in range(10):
        state = world.step_action(0.0, 0.0)
    assert math.hypot(state.pusher.x - start.x, state.pusher.y - start.y) < 1e-3


@pytest.mark.parametrize(
    "action, axis, sign",
    [((0.5, 0.0), "x", 1), ((-0.5, 0.0), "x", -1),
     ((0.0, 0.5), "y", 1), ((0.0, -0.5), "y", -1)],
)
def test_pusher_moves_in_commanded_direction(world, action, axis, sign):
    start = world.get_world_state().pusher
    for _ in range(10):
        state = world.step_action(*action)

    moved = getattr(state.pusher, axis) - getattr(start, axis)
    other = "y" if axis == "x" else "x"
    drift = getattr(state.pusher, other) - getattr(start, other)

    assert sign * moved > 0.1
    assert abs(drift) < 0.02


def test_pusher_speed_is_capped(world):
    start = world.get_world_state().pusher
    state = world.step_action(10.0, 0.0)
    dt = World.CONTROL_SUBSTEPS / World.PHYSICS_HZ
    max_distance = world.pusher.max_speed * dt
    assert 0 < state.pusher.x - start.x <= max_distance * 1.05


def test_pusher_does_not_move_block_without_contact(world):
    start = world.get_world_state().block
    for _ in range(10):
        state = world.step_action(0.5, 0.0)
    assert math.hypot(state.block.x - start.x, state.block.y - start.y) < 1e-3


def test_pusher_pushes_block_in_push_direction(env):
    start = env.world.get_world_state().block
    for _ in range(25):
        env.step((0.8, 0.0))
    block = env.world.get_world_state().block
    assert block.x - start.x > 0.1
    assert abs(block.y - start.y) < 0.1


def test_pusher_pushes_block_to_goal(env):
    goal = env.world.get_world_state().goal
    start = env.world.get_world_state().block
    start_distance = math.hypot(goal.x - start.x, goal.y - start.y)

    result = None
    while not (env.terminated or env.truncated):
        state = env.world.get_world_state()
        remaining = goal.x - state.block.x
        # Push until the block is near the goal, then stop so it can settle.
        vx = 0.8 if remaining > 0.1 else 0.0
        result = env.step((vx, 0.0))

    assert env.terminated, "episode truncated before block reached the goal"
    assert not env.truncated
    assert result.task_result.success
    assert result.task_result.distance_to_goal < start_distance
    assert result.task_result.distance_to_goal <= env.task.goal_radius
    assert env.action_count < env.max_actions


def test_pushing_away_from_goal_does_not_succeed(env):
    for _ in range(25):
        result = env.step((-0.8, 0.0))
    assert not result.task_result.success
    assert not result.task_result.inside_goal
