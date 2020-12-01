import carla
import torch
import numpy as np
import random
from collections import deque
from model import *

class RandomAgent:
    def __init__(self):
        pass

    def memorize(self, *args):
        pass

    def act(self, view):
        return random.choice(range(3))


class DQNAgent:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = MobileNetV2Encoder(in_channels=7, out_classes=3)
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
        view, action, next_view, reward, done = zip(*random.choices(self.memory, k=8))
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
        self.model.load_state_dict(torch.load(path, map_location=self.device))