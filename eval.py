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
from agent import DQNAgent, RandomAgent, ManualAgent

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)

controls = [
    carla.VehicleControl(throttle=0.4, steer=-0.2),
    carla.VehicleControl(throttle=0.4),
    carla.VehicleControl(throttle=0.4, steer=+0.2),
]

env = DrivingEnv(client)
agent = DQNAgent(num_controls=len(controls))
agent.load('checkpoint copy.pth')
# agent = ManualAgent()

for episode in range(10000000):
    view = env.reset()
    for iteration in range(10000000):
        action = agent.act(view)
        control = controls[action]

        view, reward, done = env.step(control)
        cv2.imshow('seg', env.seg_rgb)
        cv2.waitKey(1)

        if done:
            print(f'Episode {episode}, iterations: {iteration}, total_reward: {env.total_reward}')
            break