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


client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)
env = DrivingEnv(client)
agent = DQNAgent(device='cuda')
agent.load('checkpoint.pth')

loss = -1
epsilon = 0.15
epsilon_decay = 0.999
epsilon_min = 0.05
max_reward = 0

for episode in range(10000000):
    view = env.reset()
    for iteration in range(1000000):
        if random.random() < epsilon:
            action = random.choice(range(3))
        else:
            action = agent.act(view)

        steer = [-0.25, 0, 0.25][action]
        throttle = 1 if action == 1 else 0.1
        control = carla.VehicleControl(throttle=throttle, steer=steer)

        next_view, reward, done = env.step(control)
        loss = agent.memorize(view, action, next_view, reward, done)
        view = next_view
        cv2.imshow('seg', env.seg_rgb)
        cv2.waitKey(1)

        if done:
            break

    epsilon = epsilon * epsilon_decay
    epsilon = max(epsilon, epsilon_min)

    print(f'Episode: {episode}, iterations: {iteration} loss: {loss}, epsilon: {epsilon}, total_reward: {env.total_reward}')

    if (episode + 1) % 5 == 0:
        agent.save('checkpoint.pth')

    if env.total_reward > max_reward:
        max_reward = env.total_reward
        agent.save('checkpoint_max.pth')
        print(f'New max_reward: {max_reward}')