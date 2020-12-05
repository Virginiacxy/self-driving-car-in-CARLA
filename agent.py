import carla
import torch
import numpy as np
import random
import keyboard
from collections import deque
from model import *

class RandomAgent:
    def __init__(self, num_controls):
        self.num_controls = num_controls
        pass

    def memorize(self, *args):
        pass

    def act(self, view):
        return random.choice(range(self.num_controls))


class ManualAgent:
    def __init__(self):
        print('Press [left arrow] [up arrow] [right arrow] to continue')

    def memorize(self, *args):
        pass

    def act(self, view):
        while True:
            if keyboard.is_pressed('left arrow'):
                return 0
            if keyboard.is_pressed('up arrow'):
                return 1
            if keyboard.is_pressed('right arrow'):
                return 2


class DQNAgent:
    def __init__(self, num_controls, device='cuda'):
        self.device = device
        self.model = ConvEncoder(in_channels=23, out_classes=num_controls)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.gamma = 0.9
        self.memory = deque(maxlen=10000)

    def memorize(self, view, action, next_view, reward, done):
        view = torch.as_tensor(view).permute(2, 0, 1).unsqueeze_(0)
        next_view = torch.as_tensor(next_view).permute(2, 0, 1).unsqueeze_(0)
        self.memory.append((view, action, next_view, reward, done))

    def learn(self):
        if len(self.memory) < 500:
            return -1
        for _ in range(10):
            view, action, next_view, reward, done = zip(*random.choices(self.memory, k=50))
            view = torch.cat(view).to(self.device)
            next_view = torch.cat(next_view).to(self.device)
            action = torch.tensor(action)
            done = torch.tensor(done)
            reward = torch.tensor(reward).float().to(self.device)

            pred = self.model(view)[torch.arange(view.size(0)), action]
            true = reward
            with torch.no_grad():
                true[~done] += self.gamma * self.model(next_view[~done]).amax(dim=1)

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
            print(quality)
            action = quality.argmax()
        return action.item()
    
    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optim'])