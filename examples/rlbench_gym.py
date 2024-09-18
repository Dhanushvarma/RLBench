import gymnasium as gym
from gymnasium.utils.performance import benchmark_step
import rlbench
import imageio
import cv2
import numpy as np
from rlbench.backend.camera_utils import project_points_from_world_to_camera, get_transform_matrix

env = gym.make('rlbench/reach_target-vision-v0', render_mode="rgb_array")
intrinsics = env.intrinsic_matrix
extrinsics = env.extrinsic_matrix
transform = get_transform_matrix(extrinsics, intrinsics)

training_steps = 120
episode_length = 40
video_writer = imageio.get_writer('video.mp4', fps=10)
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        obs = env.reset()
    obs, reward, terminate, _, _ = env.step(env.action_space.sample())
    frame = env.render()  # Note: rendering increases step time.
    # draw a circle on the image at gripper position
    gripper_pose = obs['gripper_pose'][:3]
    goal = obs['task_low_dim_state'][:3]
    gripper_pose = gripper_pose.reshape(1, 3)
    goal = goal.reshape(1, 3)
    final_gripper = project_points_from_world_to_camera(gripper_pose, transform, 360, 640)[:, ::-1] 
    final_int_gripper = final_gripper.astype(int).squeeze()
    final_goal = project_points_from_world_to_camera(goal, transform, 360, 640)[:, ::-1]
    final_int_goal = final_goal.astype(int).squeeze()
    cv2.circle(frame, final_int_gripper, 5, (0, 255, 0), -1)
    cv2.circle(frame, final_int_goal, 5, (0, 0, 255), -1)
    video_writer.append_data(frame)
print('Done')
video_writer.close()

fps = benchmark_step(env, target_duration=10)
print(f"FPS: {fps:.2f}")
env.close()

