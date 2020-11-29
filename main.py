import sys
import glob
import os
import random
import cv2
import numpy as np
import time
import threading

try:
    sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


class DrivingEnv:

    CAM_WIDTH = 800
    CAM_HEIGHT = 600
    CAM_FOV = 110

    view = None
    done = False

    def __init__(self, client):
        self.client = client
        self.lock = threading.Lock()
        self._create_main_actors()

    def step(self, control):
        with self.lock:
            self.vehicle.apply_control(control)
        return self.view, self.done

    def reset(self):
        with self.lock:
            self.view = np.zeros((self.CAM_HEIGHT, self.CAM_WIDTH, 3))
            self.done = False
            self._destroy_main_actors()
            self._create_main_actors()
        return self.view

    def _create_main_actors(self):
        world = self.client.get_world()
        blueprint_library = world.get_blueprint_library()

        # Create vehicle
        vehicle_bp = blueprint_library.filter('model3')[0]
        vehicle_spawn_point = random.choice(world.get_map().get_spawn_points())
        self.vehicle = world.spawn_actor(vehicle_bp, vehicle_spawn_point)

        # Create camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', self.CAM_WIDTH)
        camera_bp.set_attribute('image_size_y', self.CAM_HEIGHT)
        camera_bp.set_attribute('fov', self.CAM_FOV)
        camera_spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.camera = world.spawn_actor(camera_bp, camera_spawn_point, attach_to=self.vehicle)
        self.camera.listen(self._camera_update)

        # Create collision sensor
        collision_bp = world.get_blueprint_library().find('sensor.other.collision')
        collision_spawn_point = carla.Transform(carla.Location(x=0, z=0))
        self.collision = world.spawn_actor(collision_bp, collision_spawn_point, attach_to=self.vehicle)
        self.collision.listen(self._collision_update)

        # Create lane invasion detector
        lane_bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        lane_spawn_point = carla.Transform(carla.Location(x=0, z=0))
        self.lane = world.spawn_actor(lane_bp, lane_spawn_point, attach_to=self.vehicle)
        self.lane.listen(self._lane_invasion_update)

    def _destroy_main_actors(self):
        self.vehicle.destroy()
        self.camera.destroy()
        self.collision.destroy()

    def _camera_update(self, x):
        self.view = np.array(x.raw_data).reshape(self.CAM_HEIGHT, self.CAM_WIDTH, -1)[:, :, :3]

    def _collision_update(self, event):
        self.done = True

    def _lane_invasion_update(self, event):
        print(event)


class RandomAgent:
    def __init__(self):
        pass

    def act(self, view):
        return carla.VehicleControl(
            throttle=random.random(),
            steer=random.random())


client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)

env = DrivingEnv(client)
agent = RandomAgent()


for episode in range(500):
    view = env.reset()
    while True:

        control = agent.act(view)
        view, done = env.step(control)
        if done:
            break

        cv2.imshow('', view)
        cv2.waitKey(1)