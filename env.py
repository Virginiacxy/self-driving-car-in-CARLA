import carla
import numpy as np
import random
import threading
from collections import deque

import cv2


class DrivingEnv:
    CAM_WIDTH = 64
    CAM_HEIGHT = 64
    CAM_FOV = 110

    view = None
    done = False
    invasion_count = 0

    def __init__(self, client):
        self.client = client
        self.world = self.client.get_world()

        # Change to use synchronized fixed time-step 
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.5
        self.world.apply_settings(settings)

        self._create_main_actors()

    def step(self, control):
        self.vehicle.apply_control(control)
        self.world.tick()

        # Reward
        if self.done:
            reward = -50
        else:
            reward = 1

        self.invasion_count = 0
        self.total_reward += reward
        return self._get_current_view(), reward, self.done

    def reset(self):
        self.rgb_view = np.zeros((self.CAM_HEIGHT, self.CAM_WIDTH, 3))
        self.seg_view = np.zeros((self.CAM_HEIGHT, self.CAM_WIDTH, 3))
        self.depth_view = np.zeros((self.CAM_HEIGHT, self.CAM_WIDTH, 1))
        self.locations = deque(maxlen=10)
        self.done = False
        self.total_reward = 0
        self._destroy_main_actors()
        self._create_main_actors()
        self.world.tick()
        return self._get_current_view()

    def _create_main_actors(self):
        try:
            blueprint_library = self.world.get_blueprint_library()

            # Create vehicle
            vehicle_bp = blueprint_library.filter('model3')[0]
            vehicle_spawn_point = random.choice(self.world.get_map().get_spawn_points())
            self.vehicle = self.world.spawn_actor(vehicle_bp, vehicle_spawn_point)

            # Create camera
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.CAM_WIDTH))
            camera_bp.set_attribute('image_size_y', str(self.CAM_HEIGHT))
            camera_bp.set_attribute('fov', str(self.CAM_FOV))
            camera_spawn_point = carla.Transform(carla.Location(x=2.5, z=1.5))
            self.camera = self.world.spawn_actor(camera_bp, camera_spawn_point, attach_to=self.vehicle)
            self.camera.listen(self._camera_update)

            # Create collision sensor
            collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
            collision_spawn_point = carla.Transform(carla.Location(x=0, z=0))
            self.collision = self.world.spawn_actor(collision_bp, collision_spawn_point, attach_to=self.vehicle)
            self.collision.listen(self._collision_update)

            # Create depth sensor
            depth_bp = blueprint_library.find('sensor.camera.depth')
            depth_bp.set_attribute('image_size_x', str(self.CAM_WIDTH))
            depth_bp.set_attribute('image_size_y', str(self.CAM_HEIGHT))
            depth_bp.set_attribute('fov', str(self.CAM_FOV))
            depth_spawn_point = carla.Transform(carla.Location(x=2.5, z=1.5))
            self.depth_sen = self.world.spawn_actor(depth_bp, depth_spawn_point, attach_to=self.vehicle)
            self.depth_sen.listen(self._depth_sensor_update)

            # Create segmentation sensor
            seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
            seg_bp.set_attribute('image_size_x', str(self.CAM_WIDTH))
            seg_bp.set_attribute('image_size_y', str(self.CAM_HEIGHT))
            seg_bp.set_attribute('fov', str(self.CAM_FOV))
            seg_spawn_point = carla.Transform(carla.Location(x=2.5, z=1.5))
            self.seg_sen = self.world.spawn_actor(seg_bp, seg_spawn_point, attach_to=self.vehicle)
            self.seg_sen.listen(self._segmentation_sensor_update)

            # Create lane invasion detector
            lane_bp = blueprint_library.find('sensor.other.lane_invasion')
            lane_spawn_point = carla.Transform(carla.Location(x=0, z=0))
            self.lane = self.world.spawn_actor(lane_bp, lane_spawn_point, attach_to=self.vehicle)
            self.lane.listen(self._lane_invasion_update)

            # Initialize speed
            self.vehicle.apply_control(carla.VehicleControl(throttle = 1))
            for _ in range(30):
                self.world.tick()
        except:
            self._destroy_main_actors()
            self._create_main_actors()

    def _destroy_main_actors(self):
        self.vehicle.destroy()
        self.camera.destroy()
        self.collision.destroy()
        self.depth_sen.destroy()
        self.seg_sen.destroy()
        self.lane.destroy()

    def _camera_update(self, x):
        x = np.array(x.raw_data).reshape(self.CAM_HEIGHT, self.CAM_WIDTH, -1)[:, :, :3]
        x = x.astype('float32') / 255.
        self.rgb_view = x

    def _collision_update(self, event):
        self.done = True

    def _lane_invasion_update(self, event):
        self.done = True
        # self.invasion_count += 1

    def _depth_sensor_update(self, x):
        x.convert(carla.ColorConverter.LogarithmicDepth)
        x = np.array(x.raw_data).reshape(self.CAM_HEIGHT, self.CAM_WIDTH, -1)[:, :, :1]
        x = x.astype('float32') / 255.
        self.depth_view = x
    
    def _segmentation_sensor_update(self, x):
        x.convert(carla.ColorConverter.CityScapesPalette)
        x = np.array(x.raw_data).reshape(self.CAM_HEIGHT, self.CAM_WIDTH, -1)[:, :, :3]
        x = x.astype('float32') / 255.
        self.seg_view = x

    def _get_current_view(self):
        return np.concatenate([self.rgb_view, self.depth_view, self.seg_view], axis=-1).astype('float32')