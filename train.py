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
client.load_world('Town04')

env = DrivingEnv(client)
agent = DQNAgent(device='cuda')
# agent.load('checkpoint.pth')

loss = -1
epsilon = 1
epsilon_decay = 0.998
epsilon_min = 0.02

for episode in range(10000000):
    view = env.reset()
    for iteration in range(1000000):
        if random.random() < epsilon:
            action = random.choice(range(3))
        else:
            action = agent.act(view)
        control = carla.VehicleControl(throttle=1, steer=[-1, 0, 1][action])
        next_view, reward, done = env.step(control)
        loss = agent.memorize(view, action, next_view, reward, done)
        view = next_view
        cv2.imshow('rgb', view[:, :, 0:3])
        cv2.imshow('depth', view[:, :, 3])
        cv2.imshow('seg', view[:, :, 4:7])
        cv2.waitKey(1)

        if done:
            break

    epsilon = epsilon * epsilon_decay
    epsilon = max(epsilon, epsilon_min)

    print(f'Episode: {episode}, iterations: {iteration} loss: {loss}, epsilon: {epsilon}, total_reward: {env.total_reward}')

    if (episode + 1) % 5 == 0:
        agent.save('checkpoint.pth')