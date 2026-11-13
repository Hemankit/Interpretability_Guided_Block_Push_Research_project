"""A circular planar pusher controlled through two PyBullet prismatic joints.

The fixed base -> X slider -> Y slider chain constrains the tip to the XY plane.
No position/velocity resets are performed during an active physics rollout.
"""

import math

import pybullet as p


class PlanarPusher:
    X_JOINT = 0
    Y_JOINT = 1
    TIP_LINK = 1

    def __init__(
        self,
        x: float,
        y: float,
        radius: float = 0.075,
        height: float = 0.10,
        ground_clearance: float = 0.005,
        friction: float = 0.5,
        max_speed: float = 0.8,
        max_force: float = 40.0,
        color=(0.95, 0.20, 0.18, 1.0),
        physics_client_id: int = 0,
    ):
        """Create a cylindrical tip whose only degrees of freedom are X and Y.

        All dimensional values use meters; max_speed is m/s, max_force is N.
        This is a simplified Cartesian pusher, not a full robot arm.
        """
        if radius <= 0 or height <= 0 or ground_clearance < 0:
            raise ValueError("Pusher dimensions must be valid positive values.")
        if friction < 0 or max_speed <= 0 or max_force <= 0:
            raise ValueError("Friction must be nonnegative; speed and force positive.")
        if not all(math.isfinite(float(v)) for v in (x, y)):
            raise ValueError("Initial pusher position must be finite.")

        self.radius = float(radius)
        self.height = float(height)
        self.ground_clearance = float(ground_clearance)
        self.max_speed = float(max_speed)
        self.max_force = float(max_force)
        self.physics_client_id = physics_client_id
        self.vx = 0.0
        self.vy = 0.0

        # The tip bottom stays slightly above the tabletop to avoid dragging.
        tip_center_z = self.ground_clearance + self.height / 2.0

        tip_collision = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=self.radius,
            height=self.height,
            physicsClientId=self.physics_client_id,
        )
        tip_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=self.radius,
            length=self.height,
            rgbaColor=color,
            physicsClientId=self.physics_client_id,
        )

        # Fixed world base, an invisible X carriage, then the visible Y tip.
        # Parent indices use PyBullet's convention: base=0, first link=1.
        identity = [0.0, 0.0, 0.0, 1.0]
        self.body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            basePosition=[0.0, 0.0, 0.0],
            linkMasses=[0.05, 1.0],
            linkCollisionShapeIndices=[-1, tip_collision],
            linkVisualShapeIndices=[-1, tip_visual],
            linkPositions=[[0.0, 0.0, 0.0], [0.0, 0.0, tip_center_z]],
            linkOrientations=[identity, identity],
            linkInertialFramePositions=[[0.0, 0.0, 0.0]] * 2,
            linkInertialFrameOrientations=[identity, identity],
            linkParentIndices=[0, 1],
            linkJointTypes=[p.JOINT_PRISMATIC, p.JOINT_PRISMATIC],
            linkJointAxis=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            physicsClientId=self.physics_client_id,
        )
        if self.body_id < 0 or p.getNumJoints(
            self.body_id, physicsClientId=self.physics_client_id
        ) != 2:
            raise RuntimeError("Could not create the 2-joint pusher.")

        p.changeDynamics(
            self.body_id,
            self.TIP_LINK,
            lateralFriction=float(friction),
            restitution=0.0,
            physicsClientId=self.physics_client_id,
        )
        self.reset(x, y)

    def set_motion(self, vx: float, vy: float) -> None:
        """Set the commanded XY velocity (m/s), capped at max_speed.

        NOTE: This replaces the old (forward_speed, angular_speed) interface.
        """
        vx, vy = float(vx), float(vy)
        if not (math.isfinite(vx) and math.isfinite(vy)):
            raise ValueError("Velocity commands must be finite numbers.")
        magnitude = math.hypot(vx, vy)
        if magnitude > self.max_speed:
            scale = self.max_speed / magnitude
            vx *= scale
            vy *= scale
        self.vx, self.vy = vx, vy
        self.apply_motion()

    def apply_motion(self) -> None:
        """Drive both slider motors; Bullet resolves contacts and velocities."""
        for joint, target_velocity in (
            (self.X_JOINT, self.vx),
            (self.Y_JOINT, self.vy),
        ):
            p.setJointMotorControl2(
                bodyUniqueId=self.body_id,
                jointIndex=joint,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=target_velocity,
                force=self.max_force,
                physicsClientId=self.physics_client_id,
            )

    def get_position(self) -> tuple[float, float]:
        """Return actual world position of the physical tip, not the base."""
        state = p.getLinkState(
            self.body_id,
            self.TIP_LINK,
            computeForwardKinematics=True,
            physicsClientId=self.physics_client_id,
        )
        x, y = state[4][:2]  # worldLinkFramePosition
        return float(x), float(y)

    def reset(self, x: float, y: float) -> None:
        """Reposition only at episode reset, never in the physics step loop."""
        for joint, value in ((self.X_JOINT, x), (self.Y_JOINT, y)):
            p.resetJointState(
                self.body_id,
                joint,
                targetValue=float(value),
                targetVelocity=0.0,
                physicsClientId=self.physics_client_id,
            )
        self.set_motion(0.0, 0.0)


    
