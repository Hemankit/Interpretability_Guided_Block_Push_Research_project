import pybullet as p
import numpy as np

from .pusher import PlanarPusher
from environment_generation.visual_nuisances.lighting import DEFAULT_LIGHTING, Lighting
class PusherCamera:
    def __init__(self, pusher, physics_client_id):
        self.pusher = pusher
        self.physics_client_id = physics_client_id
        self.mount_height = 3.0

    def capture(self, lighting: Lighting | None = None) -> np.ndarray:
        """Read actual pusher position

pusher.get_position()


Calculate virtual camera pose

Position + mounting offset + fixed orientation


Render the scene

PyBullet view and projection matrices


Return RGB image

NumPy array, shape (128, 128, 3)"""
        pusher_pos = self.pusher.get_position()

        x, y = self.pusher.get_position()

        cam_pos = [x, y, self.mount_height]
        cam_target = [x, y, 0.0]
        cam_up = [0.0, 1.0, 0.0]

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=cam_target,
            cameraUpVector=cam_up,
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.1,
            farVal=100.0,
        )
        # PyBullet keeps the last light settings, so always pass them
        # explicitly; otherwise a previous nuisance lighting would leak.
        lighting = lighting if lighting is not None else DEFAULT_LIGHTING
        light_kwargs = {
            "lightDirection": list(lighting.direction),
            "lightAmbientCoeff": lighting.ambient,
            "lightDiffuseCoeff": lighting.diffuse,
        }
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
    width=128,
    height=128,
    viewMatrix=view_matrix,
    projectionMatrix=projection_matrix,
    physicsClientId=self.physics_client_id,
    renderer=p.ER_TINY_RENDERER,
    **light_kwargs,
)
        # convert to np.uint8, reshape to (128, 128, 4), and then take the first three channels.
        np_img = np.array(rgb_img, dtype=np.uint8).reshape((128, 128, 4))
        return np_img[:, :, :3].copy()




        