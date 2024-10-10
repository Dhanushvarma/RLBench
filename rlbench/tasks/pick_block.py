from typing import List
import numpy as np
import math
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.dummy import Dummy
from rlbench.backend.task import Task
from rlbench.backend.conditions import GraspedCondition, DetectedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.const import colors

class PickBlock(Task):

    def init_task(self) -> None:
        self.target_block = Shape('target_block')
        self.distractor_block = Shape('distractor_block')
        self.target_blocks = [self.target_block, self.distractor_block]
        self.left_boundary = SpawnBoundary([Shape('left_boundary')])
        self.right_boundary = SpawnBoundary([Shape('right_boundary')])
        self.success_sensor = ProximitySensor('pick_block_success')
        self.register_graspable_objects(self.target_blocks)
        self.register_success_conditions([
            DetectedCondition(self.target_block, self.success_sensor, negated=False),
            GraspedCondition(self.robot.gripper, self.target_block),
        ])

    def init_episode(self, index: int) -> List[str]:
        self.variation_index = index

        target_block_color_name, target_block_rgb = colors[2] # lime
        distractor_block_color_name, distractor_block_rgb = colors[0]  # red
        self.target_block.set_color(target_block_rgb)
        self.distractor_block.set_color(distractor_block_rgb)

        self.left_boundary.clear()
        self.right_boundary.clear()

        if index == 0:
            self.left_boundary.sample(self.target_block, min_distance=0.1)
            self.right_boundary.sample(self.distractor_block, min_distance=0.1)
        else:
            self.left_boundary.sample(self.distractor_block, min_distance=0.1)
            self.right_boundary.sample(self.target_block, min_distance=0.1)

        return [f'pick up {target_block_color_name} block above the height threshold',
                f'lift {target_block_color_name} block high enough',
                f'raise {target_block_color_name} block above the specified height',
                f'elevate the {target_block_color_name} block']

    def variation_count(self) -> int:
        return 2
    
    def is_static_workspace(self) -> bool:
        return True