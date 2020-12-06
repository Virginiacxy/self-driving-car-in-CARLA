import carla
import numpy as np
import random
import threading
from collections import deque

import cv2


class DrivingEnv:
    CAM_WIDTH = 128
    CAM_HEIGHT = 128
    CAM_FOV = 120

    view = None
    done = False

    def __init__(self, client):
        self.client = client
        self.client.load_world('Town05')
        self.world = self.client.get_world()

        # Change to use synchronized fixed time-step 
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.2
        self.world.apply_settings(settings)

        self._create_main_actors()

    def step(self, control):
        self.vehicle.apply_control(control)
        self.world.tick()

        # Reward
        if self.done:
            reward = -5
        else:
            reward = 1
            # velocity = self.vehicle.get_velocity()
            # curr_location = self.vehicle.get_transform().location
            # reward = (self.vehicle_location.x - curr_location.x) ** 2 + (self.vehicle_location.y - curr_location.y) ** 2 - 0.1
            # self.vehicle_location = curr_location

        self.total_reward += reward

        return self.seg_view, reward, self.done

    def reset(self):
        self.seg_view = np.zeros((self.CAM_HEIGHT, self.CAM_WIDTH, 23), dtype=np.float32)
        self.done = False
        self.total_reward = 0
        self._destroy_main_actors()
        self._create_main_actors()
        self.world.tick()
        self.vehicle_location = self.vehicle.get_transform().location
        return self.seg_view

    def _create_main_actors(self):
        try:
            blueprint_library = self.world.get_blueprint_library()

            # Create vehicle
            vehicle_bp = blueprint_library.filter('model3')[0]
            vehicle_spawn_point = random.choice(self.world.get_map().get_spawn_points())
            self.vehicle = self.world.spawn_actor(vehicle_bp, vehicle_spawn_point)

            # Create collision sensor
            collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
            collision_spawn_point = carla.Transform(carla.Location(x=0, z=0))
            self.collision = self.world.spawn_actor(collision_bp, collision_spawn_point, attach_to=self.vehicle)
            self.collision.listen(self._collision_update)

            # Create segmentation sensor
            seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
            seg_bp.set_attribute('image_size_x', str(self.CAM_WIDTH))
            seg_bp.set_attribute('image_size_y', str(self.CAM_HEIGHT))
            seg_bp.set_attribute('fov', str(self.CAM_FOV))
            seg_spawn_point = carla.Transform(carla.Location(x=2, z=5), carla.Rotation(pitch=0))
            self.seg_sen = self.world.spawn_actor(seg_bp, seg_spawn_point, attach_to=self.vehicle)
            self.seg_sen.listen(self._segmentation_sensor_update)

            # Create lane invasion detector
            lane_bp = blueprint_library.find('sensor.other.lane_invasion')
            lane_spawn_point = carla.Transform(carla.Location(x=0, z=0))
            self.lane = self.world.spawn_actor(lane_bp, lane_spawn_point, attach_to=self.vehicle)
            self.lane.listen(self._lane_invasion_update)

            # Initialize speed
            for _ in range(60):
                self.vehicle.apply_control(carla.VehicleControl(throttle = 0.5))
                self.world.tick()
        except:
            self._destroy_main_actors()
            self._create_main_actors()

    def _destroy_main_actors(self):
        self.vehicle.destroy()
        self.collision.destroy()
        self.seg_sen.destroy()
        self.lane.destroy()

    def _collision_update(self, event):
        self.done = True

    def _lane_invasion_update(self, event):
        # self.lane_invasion = True
        # self.done = True
        pass
    
    def _segmentation_sensor_update(self, x):
        seg_idx = np.array(x.raw_data).reshape(self.CAM_HEIGHT, self.CAM_WIDTH, -1)[:, :, 2]
        seg_view = np.zeros((self.CAM_HEIGHT * self.CAM_WIDTH, 23), dtype=np.float32)
        seg_view[np.arange(self.CAM_HEIGHT * self.CAM_HEIGHT), seg_idx.reshape(-1)] = 1
        seg_view = seg_view.reshape((self.CAM_HEIGHT, self.CAM_WIDTH, 23))
        self.seg_view = seg_view

        x.convert(carla.ColorConverter.CityScapesPalette)
        seg_rgb = np.array(x.raw_data).reshape(self.CAM_HEIGHT, self.CAM_WIDTH, -1)[:, :, :3]
        seg_rgb = seg_rgb.astype('float32') / 255.
        self.seg_rgb = seg_rgb
