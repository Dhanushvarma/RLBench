from abc import abstractmethod

import numpy as np

from rlbench.action_modes.arm_action_modes import ArmActionMode, JointPosition, EndEffectorPoseViaPlanning, RelativeFrame
from rlbench.action_modes.gripper_action_modes import GripperActionMode, GripperJointPosition, Discrete
from rlbench.backend.scene import Scene


class ActionMode(object):

    def __init__(self,
                 arm_action_mode: 'ArmActionMode',
                 gripper_action_mode: 'GripperActionMode'):
        self.arm_action_mode = arm_action_mode
        self.gripper_action_mode = gripper_action_mode

    @abstractmethod
    def action(self, scene: Scene, action: np.ndarray):
        pass

    @abstractmethod
    def action_shape(self, scene: Scene):
        pass

    def action_bounds(self):
        """Returns the min and max of the action mode."""
        raise NotImplementedError('You must define your own action bounds.')


class MoveArmThenGripper(ActionMode):
    """A customizable action mode.

    The arm action is first applied, followed by the gripper action.
    """

    def action(self, scene: Scene, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])
        self.arm_action_mode.action(scene, arm_action)
        self.gripper_action_mode.action(scene, ee_action)

    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene))


# RLBench is highly customizable, in both observations and action modes.
# This can be a little daunting, so below we have defined some
# common action modes for you to choose from.

class JointPositionActionMode(ActionMode):
    """A pre-set, delta joint position action mode or arm and abs for gripper.

    Both the arm and gripper action are applied at the same time.
    """

    def __init__(self):
        super(JointPositionActionMode, self).__init__(
            JointPosition(False), GripperJointPosition(True))

    def action(self, scene: Scene, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])
        self.arm_action_mode.action_pre_step(scene, arm_action)
        self.gripper_action_mode.action_pre_step(scene, ee_action)
        scene.step()
        self.arm_action_mode.action_post_step(scene, arm_action)
        self.gripper_action_mode.action_post_step(scene, ee_action)

    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene))

    def action_bounds(self):
        """Returns the min and max of the action mode."""
        return np.array(7 * [-0.1] + [0.0]), np.array(7 * [0.1] + [0.04])

class EEFPositionActionMode(ActionMode):
    """
    A pre-set, absolute end-effector position action mode, discrete for gripper.

    The arm action is first applied, followed by the gripper action.

    IMP : end-effector position is in world frame
    """

    def __init__(self):
        super(EEFPositionActionMode, self).__init__(
            EndEffectorPoseViaPlanning(
                absolute_mode=True, frame=RelativeFrame.WORLD, collision_checking=False
            ),
            Discrete(True),
        )

    def action(self, scene: Scene, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(
            action[arm_act_size:]
        )  # NOTE : in our case ee pertains to gripper
        self.arm_action_mode.action(scene, arm_action)
        self.gripper_action_mode.action(scene, ee_action)

    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene)
        )

    def action_bounds(self, scene: Scene):
        """
        quat components are in the range [-1, 1]
        """
        lower_bound_gripper, upper_bound_gripper = (
            self.gripper_action_mode.action_bounds()
        )
        lower_bound = [
            scene._workspace_minx,
            scene._workspace_miny,
            scene._workspace_minz,
            -1,
            -1,
            -1,
            -1,
            lower_bound_gripper,
        ]
        upper_bound = [
            scene._workspace_maxx,
            scene._workspace_maxy,
            scene._workspace_maxz,
            1,
            1,
            1,
            1,
            upper_bound_gripper,
        ]

        return np.array(lower_bound), np.array(upper_bound)
