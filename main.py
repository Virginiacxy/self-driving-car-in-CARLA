from collections import deque
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
from model import MobileNetV2Encoder

class RandomAgent:
    def __init__(self):
        pass

    def memorize(self, *args):
        pass

    def act(self, view):
        return random.choice(range(3))


class DQNAgent:
    def __init__(self):
        self.device = 'cuda'
        self.model = MobileNetV2Encoder(in_channels=4, out_classes=3)
        self.model.load_pretrained_weights()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.gamma = 0.99
        self.memory = deque(maxlen=1000)

    def memorize(self, view, action, next_view, reward, done):
        view = torch.as_tensor(view).permute(2, 0, 1).unsqueeze_(0)
        next_view = torch.as_tensor(next_view).permute(2, 0, 1).unsqueeze_(0)
        self.memory.append((view, action, next_view, reward, done))
        return self.learn()

    def learn(self):
        view, action, next_view, reward, done = zip(*random.choices(self.memory, k=20))
        view = torch.cat(view).to(self.device)
        next_view = torch.cat(next_view).to(self.device)
        action = torch.tensor(action)
        done = torch.tensor(done)
        reward = torch.tensor(reward).float().to(self.device)

        pred = self.model(view)[torch.arange(view.size(0)), action]
        true = reward
        true[~done] += self.gamma * self.model(next_view[~done]).amax(dim=1).detach()

        loss = torch.nn.functional.mse_loss(pred, true)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def act(self, view):
        view = torch.as_tensor(view).permute(2, 0, 1).unsqueeze_(0).to(self.device)
        with torch.no_grad():
            self.model.eval()
            quality = self.model(view)
            action = quality.argmax()
        return action.item()
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.loada_state_dict(torch.load(path, map_location=self.device))


client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)

env = DrivingEnv(client)
agent = DQNAgent()

epsilon = 1
epsilon_decay = 0.995
epsilon_min = 0.01

for episode in range(10000000):
    view = env.reset()
    while True:
        if random.random() < epsilon:
            action = random.choice(range(3))
        else:
            action = agent.act(view)
        control = carla.VehicleControl(throttle=1, steer=[-1, 0, 1][action])
        next_view, reward, done = env.step(control)
        loss = agent.memorize(view, action, next_view, reward, done)
        view = next_view
        cv2.imshow('rgb', view[:, :, :3])
        cv2.imshow('depth', view[:, :, 3])
        cv2.waitKey(1)

        if done:
            break

    epsilon = epsilon * epsilon_decay
    epsilon = max(epsilon, epsilon_min)

    print(f'Episode: {episode}, loss: {loss}, epsilon: {epsilon}')

    if (episode + 1) % 50 == 0:
        agent.save('checkpoint.pth')