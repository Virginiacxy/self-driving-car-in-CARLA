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
        self.world = self.client.get_world()
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
        camera_spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
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
        depth_spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.depth_sen = self.world.spawn_actor(depth_bp, depth_spawn_point, attach_to=self.vehicle)

        # Create LIDAR sensor
        # lidar_cam = None
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', str(32))
        lidar_bp.set_attribute('points_per_second', str(90000))
        lidar_bp.set_attribute('rotation_frequency', str(40))
        lidar_bp.set_attribute('range', str(20))
        lidar_location = carla.Location(0, 0, 2)
        lidar_rotation = carla.Rotation(0, 0, 0)
        lidar_transform = carla.Transform(lidar_location, lidar_rotation)
        self.lidar_sen = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        self.lidar_sen.listen(
            lambda point_cloud: point_cloud.save_to_disk('tutorial/new_lidar_output/%.6d.ply' % point_cloud.frame))

        # Create lane invasion detector
        lane_bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        lane_spawn_point = carla.Transform(carla.Location(x=0, z=0))
        self.lane = self.world.spawn_actor(lane_bp, lane_spawn_point, attach_to=self.vehicle)
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

    def spawn_vehicles(self, n):
        traffic_manager = self.client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints = sorted(blueprints, key=lambda bp: bp.id)
        vehicles_list = []
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        batch = []
        for idx, transform in enumerate(spawn_points):
            if idx >= n:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, transform)
                         .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))
        for response in self.client.apply_batch_sync(batch, False):
            if response.error:
                print(response.error)
            else:
                vehicles_list.append(response.actor_id)
        # destroy vehicles
        # self.client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
