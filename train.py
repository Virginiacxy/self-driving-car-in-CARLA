import sys
import glob
import os
import random
import cv2
import numpy as np
import torch
import torchvision

try:
    sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from env import DrivingEnv
from agent import DQNAgent, RandomAgent

controls = [
    carla.VehicleControl(throttle=0.4, steer=-0.2),
    carla.VehicleControl(throttle=0.4),
    carla.VehicleControl(throttle=0.4, steer=+0.2),
]

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)
env = DrivingEnv(client)
agent = DQNAgent(num_controls=len(controls), device='cuda')
# agent.load('checkpoint copy.pth')

loss = -1
epsilon = 1
epsilon_decay = 0.997
epsilon_min = 0.02
max_reward = 0


for episode in range(10000000):
    view = env.reset()
    for iteration in range(1000):
        if random.random() < epsilon:
            action = random.choice(range(len(controls)))
        else:
            action = agent.act(view)

        control = controls[action]

        next_view, reward, done = env.step(control)
        agent.memorize(view, action, next_view, reward, done)

        cv2.imshow('seg', env.seg_rgb)
        cv2.waitKey(1)

        view = next_view

        if done:
            break

    loss = agent.learn()
    epsilon = epsilon * epsilon_decay
    epsilon = max(epsilon, epsilon_min)

    print(f'Episode: {episode}, iterations: {iteration} loss: {loss}, epsilon: {epsilon}, total_reward: {env.total_reward}')

    if (episode + 1) % 5 == 0:
        agent.save('checkpoint.pth')

    if env.total_reward >= max_reward:
        max_reward = env.total_reward
        agent.save('checkpoint_max.pth')
        print(f'New max_reward: {max_reward}')