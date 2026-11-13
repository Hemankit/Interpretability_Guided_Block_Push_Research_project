"""Minimal, reproducible PyBullet tabletop pushing world.

Physics parameters remain fixed between episodes and experiments.
"""

from dataclasses import dataclass
import math

import pybullet as p
import pybullet_data

from .pusher import PlanarPusher


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float


@dataclass(frozen=True)
class WorldState:
    pusher: Vec2
    block: Vec2
    goal: Vec2
    step: int  # number of physics substeps since reset
    block_yaw: float  # radians, rotation about the tabletop normal


class World:
    PHYSICS_HZ = 240
    CONTROL_SUBSTEPS = 12  # 20 Hz control when each action runs 12 substeps
    BLOCK_HALF_EXTENTS = (0.18, 0.18, 0.05)
    BLOCK_MASS = 1.0
    FRICTION = 0.6

    def __init__(
        self,
        pusher_position: Vec2,
        block_position: Vec2,
        goal_position: Vec2,
        gui: bool = False,
        
    ):
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        if self.physics_client < 0:
            raise RuntimeError("Could not connect to PyBullet.")
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.physics_client
        )
        p.setRealTimeSimulation(0, physicsClientId=self.physics_client)
        p.setTimeStep(1.0 / self.PHYSICS_HZ, physicsClientId=self.physics_client)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.physics_client)

        self.goal_position = goal_position
        self.step_count = 0
        self.distractor_ids: list[int] = []

        self.plane_id = p.loadURDF(
            "plane.urdf", physicsClientId=self.physics_client
        )
        p.changeDynamics(
            self.plane_id,
            -1,
            lateralFriction=self.FRICTION,
            restitution=0.0,
            physicsClientId=self.physics_client,
        )

        block_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=self.BLOCK_HALF_EXTENTS,
            physicsClientId=self.physics_client,
        )
        block_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=self.BLOCK_HALF_EXTENTS,
            rgbaColor=[0.2, 0.4, 1.0, 1.0],
            physicsClientId=self.physics_client,
        )
        self.block_id = p.createMultiBody(
            baseMass=self.BLOCK_MASS,
            baseCollisionShapeIndex=block_collision,
            baseVisualShapeIndex=block_visual,
            basePosition=[
                block_position.x,
                block_position.y,
                self.BLOCK_HALF_EXTENTS[2],
            ],
            physicsClientId=self.physics_client,
        )
        p.changeDynamics(
            self.block_id,
            -1,
            lateralFriction=self.FRICTION,
            restitution=0.0,
            physicsClientId=self.physics_client,
        )

        # No collision geometry: the green goal is a visual target only.
        goal_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.25,
            length=0.008,
            rgbaColor=[0.1, 0.8, 0.2, 1.0],
            physicsClientId=self.physics_client,
        )
        self.goal_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=goal_visual,
            basePosition=[goal_position.x, goal_position.y, 0.006],
            physicsClientId=self.physics_client,
        )

        self.pusher = PlanarPusher(
            x=pusher_position.x,
            y=pusher_position.y,
            physics_client_id=self.physics_client,
        )
        self._initial_pusher = pusher_position
        self._initial_block = block_position
        self._initial_goal = goal_position
        self._default_rgba = {
            body_id: self._read_rgba(body_id)
            for body_id in (self.block_id, self.plane_id)
        }

    def _read_rgba(self, body_id: int) -> tuple[float, ...]:
        data = p.getVisualShapeData(
            body_id, physicsClientId=self.physics_client
        )
        return tuple(data[0][7])

    def restore_default_visuals(self) -> None:
        """Return every visual nuisance target to its construction-time state."""
        for body_id, rgba in self._default_rgba.items():
            p.changeVisualShape(
                body_id,
                -1,
                rgbaColor=rgba,
                physicsClientId=self.physics_client,
            )
        for body_id in self.distractor_ids:
            p.removeBody(body_id, physicsClientId=self.physics_client)
        self.distractor_ids = []

    def step_simulation(self, steps: int = 1) -> None:
        """Perform an integer number of fixed-duration physics steps."""
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        self.pusher.apply_motion()
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.physics_client)
            self.step_count += 1

    def step_action(
        self, vx: float, vy: float, control_substeps: int = CONTROL_SUBSTEPS
    ) -> WorldState:
        """Apply XY velocity for one control interval and return the new state."""
        self.pusher.set_motion(vx, vy)
        self.step_simulation(control_substeps)
        # Stop the pusher between calls; next action can immediately override.
        self.pusher.set_motion(0.0, 0.0)
        return self.get_world_state()

    def reset(
        self,
        pusher_position: Vec2 | None = None,
        block_position: Vec2 | None = None,
        goal_position: Vec2 | None = None,
        block_yaw: float = 0.0,
    ) -> WorldState:
        """Reset state without reconstructing the world. Call between episodes.

        Defaults reproduce the constructor's starting configuration.
        """
        pusher_position = pusher_position or self._initial_pusher
        block_position = block_position or self._initial_block
        goal_position = goal_position or self._initial_goal
        if not math.isfinite(float(block_yaw)):
            raise ValueError("block_yaw must be finite")

        half_height = self.BLOCK_HALF_EXTENTS[2]
        p.resetBasePositionAndOrientation(
            self.block_id,
            [block_position.x, block_position.y, half_height],
            p.getQuaternionFromEuler([0.0, 0.0, float(block_yaw)]),
            physicsClientId=self.physics_client,
        )
        p.resetBaseVelocity(
            self.block_id,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=self.physics_client,
        )
        self.pusher.reset(pusher_position.x, pusher_position.y)
        p.resetBasePositionAndOrientation(
            self.goal_id,
            [goal_position.x, goal_position.y, 0.006],
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.physics_client,
        )
        self.goal_position = goal_position
        self.step_count = 0
        return self.get_world_state()

    def get_world_state(self) -> WorldState:
        x, y = self.pusher.get_position()
        block_pos, block_orn = p.getBasePositionAndOrientation(
            self.block_id, physicsClientId=self.physics_client
        )
        yaw = p.getEulerFromQuaternion(block_orn)[2]
        return WorldState(
            pusher=Vec2(x, y),
            block=Vec2(float(block_pos[0]), float(block_pos[1])),
            goal=self.goal_position,
            step=self.step_count,
            block_yaw=float(yaw),
        )

    def close(self) -> None:
        if p.isConnected(self.physics_client):
            p.disconnect(physicsClientId=self.physics_client)
