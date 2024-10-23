from typing import List
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from pyrep.objects.dummy import Dummy
from rlbench.backend.conditions import GraspedCondition, DetectedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.const import colors


class PickBlockEasy(Task):

    def init_task(self) -> None:
        self.green_block = Shape("target_block")
        self.red_block = Shape("distractor_block")  # ignore bad naming
        self.target_blocks = [self.green_block, self.red_block]
        self.left_boundary = SpawnBoundary([Shape("left_boundary")])
        self.right_boundary = SpawnBoundary([Shape("right_boundary")])
        self.success_sensor = ProximitySensor("pick_block_success")

        self.target_block_index = None
        self.target_block = None

        self.register_graspable_objects(self.target_blocks)
        self.register_waypoint_ability_start(0, self._move_above_next_target)

    def init_episode(self, index: int) -> List[str]:
        self.variation_index = index

        # colors
        green_color_name, green_rgb = colors[2]  # lime
        red_color_name, red_rgb = colors[0]
        _color_names = [green_color_name, red_color_name]

        # setting colors
        self.green_block.set_color(green_rgb)
        self.red_block.set_color(red_rgb)

        # clear boundaries
        self.left_boundary.clear()
        self.right_boundary.clear()

        # sample boundaries
        self.left_boundary.sample(self.green_block, min_distance=0.1)
        self.right_boundary.sample(self.red_block, min_distance=0.1)

        # assign target and distractor
        self.target_block_index = index
        self.target_block = self.target_blocks[index]

        # success condition
        self.register_success_conditions(
            [
                DetectedCondition(
                    self.target_block, self.success_sensor, negated=False
                ),
                GraspedCondition(self.robot.gripper, self.target_block),
            ]
        )

        return [
            f"pick up {_color_names[index]} block above the height threshold",
            f"lift {_color_names[index]} block high enough",
            f"raise {_color_names[index]} block above the specified height",
            f"elevate the {_color_names[index]} block",
        ]

    def _move_above_next_target(self, _):
        w2 = Dummy("waypoint1")
        x, y, z = self.target_blocks[self.target_block_index].get_position()
        _, _, oz = self.target_blocks[self.target_block_index].get_orientation()
        ox, oy, _ = w2.get_orientation()
        w2.set_position([x, y, z])
        w2.set_orientation([ox, oy, -oz])

    def variation_count(self) -> int:
        return 2

    def is_static_workspace(self) -> bool:
        return True
