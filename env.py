import carla
import numpy as np
import random
import threading

class DrivingEnv:

    CAM_WIDTH = 224
    CAM_HEIGHT = 224
    CAM_FOV = 110

    view = None
    done = False
    invasion_count = 0

    def __init__(self, client):
        self.client = client
        self.lock = threading.Lock()
        self._create_main_actors()

    def step(self, control):
        with self.lock:
            self.vehicle.apply_control(control)

            # Reward
            if self.done:
                reward = -50
            else:
                reward = 1 - 10 * self.invasion_count
            self.invasion_count = 0
        return self.view, reward, self.done

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
        camera_bp.set_attribute('image_size_x', str(self.CAM_WIDTH))
        camera_bp.set_attribute('image_size_y', str(self.CAM_HEIGHT))
        camera_bp.set_attribute('fov', str(self.CAM_FOV))
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
        with self.lock:
            self.view = np.array(x.raw_data).reshape(self.CAM_HEIGHT, self.CAM_WIDTH, -1)[:, :, :3]

    def _collision_update(self, event):
        with self.lock:
            self.done = True

    def _lane_invasion_update(self, event):
        with self.lock:
            self.invasion_count += 1
