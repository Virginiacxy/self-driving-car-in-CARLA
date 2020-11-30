import glob
import os
import sys
try:
    sys.path.append('/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg')
except IndexError:
    pass
import carla

import random
import time
import numpy as np
import cv2

IM_WIDTH = 640
IM_HEIGHT = 480

# actor_list = []
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(1.0)

    blueprints = world.get_blueprint_library().filter('vehicle.*')
    blueprints = sorted(blueprints, key=lambda bp: bp.id)
    vehicles_list = []
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    SetVehicleLightState = carla.command.SetVehicleLightState
    FutureActor = carla.command.FutureActor
    batch = []
    for n, transform in enumerate(spawn_points):
        if n >= 10:
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
    for response in client.apply_batch_sync(batch, False):
        if response.error:
            print(response.error)
        else:
            vehicles_list.append(response.actor_id)
    time.sleep(100)
finally:
    print('destroying actors')
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
    print('done.')