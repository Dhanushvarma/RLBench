import argparse
import os
import pickle
from multiprocessing import Manager, Process

import numpy as np
from pyrep.const import RenderMode

import rlbench.backend.task as task
from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import EEFPositionActionMode, MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import (
    EndEffectorPoseViaIK,
    EndEffectorPoseViaPlanning,
    JointPosition,
    JointVelocity,
)
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import utils
from rlbench.backend.const import *
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment


def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_demo(demo, example_path):
    # Save the low-dimension data
    with open(os.path.join(example_path, LOW_DIM_PICKLE), "wb") as f:
        pickle.dump(demo, f)


def run(i, lock, task_index, variation_count, results, file_lock, tasks, args):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    # Initialise each thread with random seed
    np.random.seed(None)
    num_tasks = len(tasks)

    # only proprioceptive and low-dim
    obs_config = ObservationConfig()
    obs_config.set_all(False)  # Disable all observations
    obs_config.joint_positions = True
    obs_config.joint_velocities = True
    obs_config.joint_forces = True
    obs_config.gripper_open = True
    obs_config.gripper_pose = True
    obs_config.gripper_joint_positions = True
    obs_config.gripper_touch_forces = True
    obs_config.task_low_dim_state = True

    # rlbench_env = Environment(
    #     action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
    #     obs_config=obs_config,
    #     arm_max_velocity=args.arm_max_velocity,
    #     arm_max_acceleration=args.arm_max_acceleration,
    #     headless=True,
    # )

    if args.arm_action_mode == "joint_position":
        arm_action_mode = JointPosition()
    elif args.arm_action_mode == "joint_velocity":
        arm_action_mode = JointVelocity()
    elif args.arm_action_mode == "eep":
        arm_action_mode = EndEffectorPoseViaPlanning(absolute_mode=True)
    else:
        raise NotImplementedError(
            f"Arm action mode {args.arm_action_mode} not implemented."
        )

    print("Arm action mode: ", args.arm_action_mode)
    rlbench_env = Environment(
        action_mode=EEFPositionActionMode(),
        # action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        arm_max_velocity=args.arm_max_velocity,
        arm_max_acceleration=args.arm_max_acceleration,
        headless=True,
    )
    rlbench_env.launch()

    args.save_path = args.save_path + "_lowdim_" + args.arm_action_mode

    task_env = None

    tasks_with_problems = results[i] = ""

    while True:
        # Figure out what task/variation this thread is going to do
        with lock:
            if task_index.value >= num_tasks:
                print("Process", i, "finished")
                break

            my_variation_count = variation_count.value
            t = tasks[task_index.value]
            task_env = rlbench_env.get_task(t)
            var_target = task_env.variation_count()
            if args.variations >= 0:
                var_target = np.minimum(args.variations, var_target)
            if my_variation_count >= var_target:
                # If we have reached the required number of variations for this
                # task, then move on to the next task.
                variation_count.value = my_variation_count = 0
                task_index.value += 1

            variation_count.value += 1
            if task_index.value >= num_tasks:
                print("Process", i, "finished")
                break
            t = tasks[task_index.value]

        task_env = rlbench_env.get_task(t)
        task_env.set_variation(my_variation_count)
        descriptions, _ = task_env.reset()

        variation_path = os.path.join(
            args.save_path, task_env.get_name(), VARIATIONS_FOLDER % my_variation_count
        )

        check_and_make(variation_path)

        with open(os.path.join(variation_path, VARIATION_DESCRIPTIONS), "wb") as f:
            pickle.dump(descriptions, f)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        check_and_make(episodes_path)

        abort_variation = False
        for ex_idx in range(args.episodes_per_task):
            print(
                "Process",
                i,
                "// Task:",
                task_env.get_name(),
                "// Variation:",
                my_variation_count,
                "// Demo:",
                ex_idx,
            )
            attempts = 10
            while attempts > 0:
                try:
                    # TODO: for now we do the explicit looping.
                    (demo,) = task_env.get_demos(amount=1, live_demos=True)
                except Exception as e:
                    attempts -= 1
                    if attempts > 0:
                        continue
                    problem = (
                        "Process %d failed collecting task %s (variation: %d, "
                        "example: %d). Skipping this task/variation.\n%s\n"
                        % (i, task_env.get_name(), my_variation_count, ex_idx, str(e))
                    )
                    print(problem)
                    tasks_with_problems += problem
                    abort_variation = True
                    break
                episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)
                check_and_make(episode_path)
                with file_lock:
                    save_demo(demo, episode_path)
                break
            if abort_variation:
                break

    results[i] = tasks_with_problems
    rlbench_env.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(description="RLBench Dataset Generator")
    parser.add_argument(
        "--save_path",
        type=str,
        default="/tmp/rlbench_data/",
        help="Where to save the demos.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=[],
        help="The tasks to collect. If empty, all tasks are collected.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="The number of parallel processes during collection.",
    )
    parser.add_argument(
        "--episodes_per_task",
        type=int,
        default=10,
        help="The number of episodes to collect per task.",
    )
    parser.add_argument(
        "--variations",
        type=int,
        default=-1,
        help="Number of variations to collect per task. -1 for all.",
    )
    parser.add_argument(
        "--arm_action_mode",
        type=str,
        default="joint_position",
        help="arm action mode for the environment.",
    )
    parser.add_argument(
        "--arm_max_velocity",
        type=float,
        default=1.0,
        help="Max arm velocity used for motion planning.",
    )
    parser.add_argument(
        "--arm_max_acceleration",
        type=float,
        default=4.0,
        help="Max arm acceleration used for motion planning.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    task_files = [
        t.replace(".py", "")
        for t in os.listdir(task.TASKS_PATH)
        if t != "__init__.py" and t.endswith(".py")
    ]

    if len(args.tasks) > 0:
        for t in args.tasks:
            if t not in task_files:
                raise ValueError("Task %s not recognised!." % t)
        task_files = args.tasks

    tasks = [task_file_to_task_class(t) for t in task_files]

    manager = Manager()

    result_dict = manager.dict()
    file_lock = manager.Lock()

    task_index = manager.Value("i", 0)
    variation_count = manager.Value("i", 0)
    lock = manager.Lock()

    check_and_make(args.save_path)

    processes = [
        Process(
            target=run,
            args=(
                i,
                lock,
                task_index,
                variation_count,
                result_dict,
                file_lock,
                tasks,
                args,
            ),
        )
        for i in range(args.processes)
    ]
    [t.start() for t in processes]
    [t.join() for t in processes]

    print("Data collection done!")
    for i in range(args.processes):
        print(result_dict[i])


if __name__ == "__main__":
    main()
