import sys
import glob
import os

try:
    sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import cv2

from env import DrivingEnv
from agent import DQNAgent, RandomAgent

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)

env = DrivingEnv(client)
agent = DQNAgent()
agent.load('checkpoint.pth')

for episode in range(10000000):
    view = env.reset()
    while True:
        action = agent.act(view)
        control = carla.VehicleControl(throttle=1, steer=[-1, 0, 1][action])
        view, reward, done = env.step(control)
        cv2.imshow('rgb', view[:, :, 0:3])
        cv2.imshow('depth', view[:, :, 3])
        cv2.imshow('seg', view[:, :, 4:7])
        cv2.waitKey(1)

        if done:
            print(f'Episode {episode}, total_reward: {env.total_reward}')
            break