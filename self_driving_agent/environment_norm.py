import glob
import os
import sys
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time

import logging
from torch.utils.tensorboard import SummaryWriter


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import pickle

from synch_mode import CarlaSyncMode
from controllers import PIDLongitudinalController
from utils import *
from settings import *
from DDPG.ddpg_torch import *
from DDPG_parameters import *
from config import config

random.seed(78)

class SimEnv(object):
    def __init__(self, visuals=True, target_speed = 30, max_iter = 4000, start_buffer = 10, train_freq = 1,
        save_freq = 200, start_ep = 0, max_dist_from_waypoint = 20) -> None:
        

        self.visuals = visuals
        if self.visuals:
            self._initiate_visuals()

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)

        self.world = self.client.load_world('Town02_Opt')
        self.world.unload_map_layer(carla.MapLayer.Decals)
        self.world.unload_map_layer(carla.MapLayer.Foliage)
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        self.world.unload_map_layer(carla.MapLayer.Particles)
        self.world.unload_map_layer(carla.MapLayer.Props)
        self.world.unload_map_layer(carla.MapLayer.StreetLights)
        

        self.spawn_points = self.world.get_map().get_spawn_points()

        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_blueprint = self.blueprint_library.find('vehicle.nissan.patrol')
        #self.vehicle_blueprint = self.blueprint_library.find('vehicle.tesla.model3')

        # input these later on as arguments
        self.global_t = 0 # global timestep
        self.target_speed = target_speed # km/h 
        self.max_iter = max_iter
        self.start_buffer = start_buffer
        self.train_freq = train_freq
        self.save_freq = save_freq
        self.start_ep = start_ep

        self.max_dist_from_waypoint = max_dist_from_waypoint
        self.start_train = self.start_ep + self.start_buffer
        
        self.total_rewards = 0
        self.average_rewards_list = []

        # Additional attributes for yaw calculation
        self.previous_yaw = None
        self.previous_yaw_rate = None
        self.delta_time = 1.0 / 30  # Assuming 30 FPS, adjust based on your simulation setup

        # Initiate states
        self.initial_observations = []
        self.navigation_obs = []

        # max_size: memory size for the replay buffer
        #

        # Initialize DDPG agent
        self.agent = DDPGAgent(alpha=LR_ACTOR, beta=LR_CRITIC, 
                          input_dims=INPUT_DIMENSION, 
                          tau=TAU, env=None, gamma=GAMMA,
                          n_actions=1, max_size=BUFFER_SIZE, 
                          layer1_size=LAYER1_SIZE, layer2_size=LAYER2_SIZE,
                          batch_size=BATCH_SIZE)
        
        # Initialize state vectors for normalization
        self.state_data = []  # List to collect state data for normalization
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Min-max scaler for normalization
        self.scaler_fitted = False  # Flag to check if the scaler is fitted

        # # Initialize logging and TensorBoard writer
        # self.logger = logging.getLogger(self.__class__.__name__)
        # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        # self.writer = SummaryWriter(log_dir='./logs/simenv')
        log_dir = './logs/simenv'
        log_file = os.path.join(log_dir, 'simulation.log')

        csv_file = './csv/simenv.csv'
        csv_file_norm = './csv/simenv_norm.csv'

        self.logger = initialize_logger(log_file)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.csv_writer, self.csv_file_handle = initialize_csv_writer(csv_file)
        self.csv_writer_norm, self.csv_file_handle_norm = initialize_csv_writer(csv_file_norm)
        

    def _initiate_visuals(self):
        pygame.init()

        self.display = pygame.display.set_mode(
            (800, 600),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.font = get_font()
        self.clock = pygame.time.Clock()
    
    def create_actors(self):
        self.actor_list = []
        # spawn vehicle at random location
        self.vehicle = self.world.spawn_actor(self.vehicle_blueprint, random.choice(self.spawn_points))
        # vehicle.set_autopilot(True)
        self.actor_list.append(self.vehicle)

        self.camera_rgb = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_rgb)

        self.camera_rgb_vis = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_rgb_vis)

        self.collision_sensor = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.collision'),
            carla.Transform(),
            attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)

        self.speed_controller = PIDLongitudinalController(self.vehicle)

        # INITIATE STATE VALUES
        self.throttle = float(0.0)
        self.previous_steer = float(0.0)
        self.velocity = float(0.0)
        self.velocity_x = float(0.0)
        self.velocity_y = float(0.0)
        self.velocity_z = float(0.0)
        self.distance_from_center = float(0.0)
        self.angle = float(0.0)
        self.center_lane_deviation = 0.0
        self.distance_covered = 0.0
        self.euclidean_dist_list = []
        self.dev_angle_array = []
        self.engine_rpm = float(0.0)
    
    def reset(self):
        for actor in self.actor_list:
            actor.destroy()
   
    def generate_episode(self, ep, eval=True):
        with CarlaSyncMode(self.world, self.camera_rgb, self.camera_rgb_vis, self.collision_sensor, fps=30) as sync_mode:
            counter = 0

            #snapshot, image_rgb, image_rgb_vis, collision = sync_mode.tick(timeout=2.0)
            # snapshot, collision = sync_mode.tick(timeout=2.0) # Snapshot is a carla.WorldSnapshot

            returned_data = sync_mode.tick(timeout=2.0)
            # Assuming the first item in returned_data is the snapshot
            snapshot = returned_data[0]
            # Assuming the last item is the collision data
            collision = returned_data[-1]
            # Retrieving image_rgb_vis, which should be the second-to-last item
            image_rgb_vis = returned_data[-2]

            # destroy if there is no data
            #if snapshot is None or image_rgb is None:
            if snapshot is None:
                print("No data, skipping episode")
                self.reset()
                return None
            
            # Generate route
            self.route_waypoints = []
            self.current_waypoint_index = 0
            self.total_distance = 780 # depending on the town

            # Get the initial waypoint based on the vehicle's current location
            current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location(), 
                                                        project_to_road=True, lane_type=carla.LaneType.Driving)
            self.route_waypoints.append(current_waypoint)

            for x in range(self.total_distance):
                if x < 650:
                    next_waypoint = current_waypoint.next(5.0)[0]
                else:
                    next_waypoint = current_waypoint.next(5.0)[-1]

                self.route_waypoints.append(next_waypoint)
                current_waypoint = next_waypoint


            try:
                # Image processing removed as it's not needed
                #image = process_img(image_rgb)
                #next_state = image 
                initial_observations = np.array([*self.euclidean_dist_list, *self.dev_angle_array,
                                                self.velocity, self.velocity_x, self.velocity_y,self.velocity_z,
                                                self.engine_rpm, self.distance_from_center, self.angle])
                next_state = initial_observations

                # # TO DRAW WAYPOINTS
                # for w in self.route_waypoints:
                #     self.world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
                #                                 color=carla.Color(r=0, g=0, b=255), life_time=120.0,
                #                                 persistent_lines=True)

                while True:
                    if self.visuals: # Check the visuals and maintain a consistent frame rate
                        if should_quit(): # utils.py, checks pygame for quit events
                            return
                        self.clock.tick_busy_loop(30) # does not advance the simulation, only controls the fps rate

                    #vehicle_location = self.vehicle.get_location()
                    # Location of the car
                    self.location = self.vehicle.get_location()

                    waypoint = self.world.get_map().get_waypoint(self.location, project_to_road=True, 
                       lane_type=carla.LaneType.Driving)
                    
                    
                    #speed = get_speed(self.vehicle)

                    # Get the new states
                    # Retrieve the velocity of the vehicle
                    # and calculate it in km/h
                    velocity = self.vehicle.get_velocity() # Return: carla.Vector3D - m/s
                    self.velocity_x = velocity.x * 3.6
                    self.velocity_y = velocity.y * 3.6
                    self.velocity_z = velocity.z * 3.6
                    self.velocity = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6

                    # Estimate the engine_rpm
                    self.engine_rpm = self.estimate_engine_rpm()

                    # Estimate the wheel_rpm
                    self.wheel_rpm = self.estimate_wheel_rpm(velocity)
                    
                    
                    # Rotation of the vehicle in correlation to the map/lane
                    # get_transform(self): Returns the actor's transform (location and rotation)
                    # the client received during last tick. The method does not call the simulator
                    # rotation (carla.Rotation - degrees (pitch, yaw, roll))
                    #self.rotation = self.vehicle.get_transform().rotation.yaw 

                    # Get the current yaw from the vehicle's transform
                    current_yaw = self.vehicle.get_transform().rotation.yaw

                    # Calculate current yaw rate and yaw acceleration
                    current_yaw_rate = self.calculate_yaw_rate(current_yaw)
                    yaw_acceleration = self.calculate_yaw_acceleration(current_yaw_rate)

                    #transform = self.vehicle.get_transform()
                    # Keep track of closest waypoint on the route
                    waypoint_index = self.current_waypoint_index

                    # The purpose of this for loop is to determine 
                    # if a vehicle has passed the next waypoint along a predefined route
                    for _ in range(len(self.route_waypoints)):
                        # Check if we passed the next waypoint along the route
                        next_waypoint_index = waypoint_index + 1
                        wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]

                        # Computes the dot product between the 2D projection 
                        # of the forward vector of the waypoint (wp.transform.get_forward_vector())
                        # and the vector from the current vehicle location to the waypoint (self.location - wp.transform.location):
                        # positive dot product -> the vectors are pointing at the same direction
                        dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],self.vector(self.location - wp.transform.location)[:2])
                        
                        # If the dot product is positive, 
                        # it means the vehicle has passed the next waypoint along the route
                        # else, it breaks out of the loop, 
                        # indicating that the current waypoint has not been passed yet:
                        if dot > 0.0:
                            waypoint_index += 1
                        else:
                            break
                    
                    # waypoint_index: the calculated waypoint index, 
                    # which represents the closest waypoint that the vehicle has passed along the planned route
                    self.current_waypoint_index = waypoint_index
                    # Calculate deviation from center of the lane
                    # % isto ensure that we remain within the limits of the route points
                    self.current_waypoint = self.route_waypoints[ self.current_waypoint_index    % len(self.route_waypoints)]
                    self.next_waypoint = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
                    
                    # CALCULATE d (distance_from_center) and Theta (angle)
                    # The result is the distance of the vehicle from the center of the lane:
                    self.distance_from_center = self.distance_to_line(self.vector(self.current_waypoint.transform.location),self.vector(self.next_waypoint.transform.location),self.vector(self.location))
                    # Get angle difference between closest waypoint and vehicle forward vector
                    fwd    = self.vector(self.vehicle.get_velocity())
                    wp_fwd = self.vector(self.current_waypoint.transform.rotation.get_forward_vector()) # Return: carla.Vector3D
                    self.angle  = self.angle_diff(fwd, wp_fwd)
                    
                    # Calculate calculate_euc_dist
                    self.euclidean_dist_list = self.calculate_euc_dist(self.current_waypoint_index)

                    # Calculate dev_angle_list
                    self.dev_angle_array = self.calculate_deviation_angle_tan(self.current_waypoint_index)



                    # # Print states:
                    # print(f"Velocity (x): {self.velocity_x:.2f} km/h")
                    # print(f"Velocity (y): {self.velocity_y:.2f} km/h")
                    # print(f"Velocity (z): {self.velocity_z:.2f} km/h")
                    # print(f"Total Velocity: {self.velocity:.2f} km/h")
                    # print(f"Estimated Engine RPM: {self.engine_rpm:.2f} rpm")
                    # print(f"Distance to line: {self.distance_from_center:.2f} unit")
                    # # Assuming self.euclidean_dist_list is a list of numbers
                    # formatted_dist_list = [f"{dist:.2f}" for dist in self.euclidean_dist_list]
                    # formatted_dist_str = ", ".join(formatted_dist_list)
                    # print(f"Euclidean Distance list: {formatted_dist_str} unit")
                    # # Assuming self.dev_angle_array is a list of numbers
                    # formatted_dev_angle_array = [f"{dist:.2f}" for dist in self.dev_angle_array]
                    # formatted_dev_angle_str = ", ".join(formatted_dev_angle_array)
                    # print(f"Deviation Angle list: {formatted_dev_angle_str} unit")
                    # print(f"Angle Diff: {self.angle:.2f} unit")

                    # print("Length of euclidean_dist_array:", len(self.euclidean_dist_list))
                    # print("Length of dev_angle_array:", len(self.dev_angle_array))

                    # Logging 
                    self.logger.info(f"Episode: {ep}, Distance from Center: {self.distance_from_center}, Angle Deviation: {self.angle}")
                    self.writer.add_scalar('Performance/Distance from Center', self.distance_from_center, ep)
                    self.writer.add_scalar('Performance/Angle Deviation', self.angle, ep)


                    new_observations = np.array([*self.euclidean_dist_list, *self.dev_angle_array,
                                     self.velocity, self.velocity_x, self.velocity_y, self.velocity_z,
                                     self.engine_rpm, self.distance_from_center, self.angle])
                    
                    
                    # print("Total state length:", len(new_observations))
                    
                    # Advance the simulation and wait for the data.
                    state = new_observations

                    counter += 1
                    self.global_t += 1


                    # # action = model.select_action(state, eval=eval)
                    action = self.agent.choose_action(state)
                    steer_nd = action
                    steer = steer_nd.item()

                    # print("steer_nd:", steer_nd)
                    # print("steer_nd type:", type(steer_nd))
                    # print("steer:", steer)
                    # print("steer type:", type(steer))

                    control = self.speed_controller.run_step(self.target_speed)
                    control.steer = steer
                    self.vehicle.apply_control(control)

                    # print("snapshot:", snapshot)
                    # print("snapshot type:", type(snapshot))

                    fps = round(1.0 / snapshot.timestamp.delta_seconds)

                    # NEXT TICK
                    #snapshot, image_rgb, image_rgb_vis, collision = sync_mode.tick(timeout=2.0)
                    #snapshot, collision = sync_mode.tick(timeout=2.0)

                    returned_data = sync_mode.tick(timeout=2.0)
                    # Assuming the first item in returned_data is the snapshot
                    snapshot = returned_data[0]
                    # Assuming the last item is the collision data
                    collision = returned_data[-1]
                    # Retrieving image_rgb_vis, which should be the second-to-last item
                    image_rgb_vis = returned_data[-2]

                    # OLD REWARD CALL
                    cos_yaw_diff, dist, collision = get_reward_comp(self.vehicle, waypoint, collision)
                    # reward = reward_value(cos_yaw_diff, dist, collision)

                    # UPDATED REWARD
                    # reward = calculate_reward_1(self.angle, self.distance_from_center, self.velocity, yaw_acceleration)
                    reward = calculate_reward_A2(velocity.x, steer, collision, self.angle)

                    #if snapshot is None or image_rgb is None:
                    if snapshot is None:
                        print("Process ended here")
                        break

                    #image = process_img(image_rgb)

                    done = 1 if collision else 0

                    self.total_rewards += reward

                    next_state = state

                    # Logging state before normalization
                    self.logger.info(f"Pre-Norm State: {state}")
                    self.csv_writer.writerow([time.time(), ep, *state, steer, reward, self.total_rewards])

                    ###################
                    # Normalization logic
                    ###################
                    if ep <= 100:
                        # Collect state data for normalization
                        self.state_data.append(state)
                    else: 
                        if not self.scaler_fitted:
                            # Fit the scaler to the collected state data
                            self.scaler.fit(self.state_data)
                            self.scaler_fitted = True # to ensure the one-time fitting of the scaler

                    # Check if the scaler is fitted, then normalize the state
                    if self.scaler_fitted:
                        state = self.normalize_state(state)
                        self.logger.info(f"Post-Norm State: {state}")
                        # self.csv_writer_norm.writerow([time.time(), ep, *self.euclidean_dist_list, *self.dev_angle_array,
                        #   self.velocity, self.velocity_x, self.velocity_y, self.velocity_z,
                        #   self.engine_rpm, self.distance_from_center, self.angle])
                        self.csv_writer_norm.writerow([time.time(), ep, *state, steer, reward, self.total_rewards])



                    #CHECK THIS
                    #replay_buffer.add(state, action, next_state, reward, done)
                    self.agent.remember(state, action, reward, next_state, done)

                    if not eval:
                        # Train after a number of episodes > start_train and do not train with every timestep
                        # if ep > self.start_train and (self.global_t % self.train_freq) == 0:
                        if ep > 100 and (self.global_t % self.train_freq) == 0:
                            #model.train(replay_buffer)
                            self.agent.learn()

                    # Draw the display.
                    if self.visuals:
                        draw_image(self.display, image_rgb_vis)
                        self.display.blit(
                            self.font.render('% 5d FPS (real)' % self.clock.get_fps(), True, (255, 255, 255)),
                            (8, 10))
                        self.display.blit(
                            self.font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                            (8, 28))
                        pygame.display.flip()

                    if collision == 1 or counter >= self.max_iter or dist > self.max_dist_from_waypoint:
                        print("Episode {} processed".format(ep), counter)
                        break
                
                if ep % self.save_freq == 0 and ep > 0:
                    self.save(ep)
                    

                # Logging
                print("Episode {} total rewards".format(ep), self.total_rewards)
                self.logger.info(f"Episode: {ep}, Total Reward: {self.total_rewards}")
                self.writer.add_scalar('Rewards/Total Reward', self.total_rewards, ep)


            except KeyboardInterrupt:
                print("Simulation stopped by the user.")
                sys.exit()  # This will terminate the program

    def save(self, ep):
        if ep % self.save_freq == 0 and ep > self.start_ep:
            avg_reward = self.total_rewards/self.save_freq
            self.average_rewards_list.append(avg_reward)
            self.total_rewards = 0

            #model.save('weights/model_ep_{}'.format(ep))
            self.agent.save_models()
            print("Saved model with average reward =", avg_reward)
    
    def quit(self):
        pygame.quit()
        # End logging
        self.writer.close()
        self.agent.writer.close()
        self.csv_file_handle.close()
        self.csv_file_handle_norm.close()

    # -------------------------------------------------
    # Estimating Engine RPM                          |
    # -------------------------------------------------
    def estimate_engine_rpm(self):
        # Get the control input for the vehicle
        control = self.vehicle.get_control()
        physics = self.vehicle.get_physics_control()

        # Calculate engine RPM
        engine_rpm = physics.max_rpm * control.throttle

        # Throttle values: [0.0, 1.0]. Default is 0.0.
        # Max rpm values: For most passenger cars, a max RPM in the range of 6,000 to 8,000 RPM.

        # Check if the vehicle is in gear (not in neutral or reverse)
        if control.gear > 0:
            # Retrieve the gear information for the current gear
            gear = physics.forward_gears[control.gear]

            # Adjust engine RPM based on the gear ratio of the current gear
            engine_rpm *= gear.ratio

        return engine_rpm

    # -------------------------------------------------
    # Estimating Engine RPM                          |
    # -------------------------------------------------
    def estimate_wheel_rpm(self, velocity):
        # Get the control input for the vehicle
        physics = self.vehicle.get_physics_control()

        # Retrieve radius
        radius_cm = physics.wheels[0].radius
        radius_m = radius_cm/100

        # Calculate the wheel circumference in meters
        wheel_circumference_m = 2 * math.pi * radius_m

        # Convert the vehicle's velocity vector to speed in m/s
        speed_m_s = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    
        # Calculate wheel rotation in revolutions per second (RPS)
        wheel_rps = speed_m_s / wheel_circumference_m

        # Convert RPS to revolutions per minute (RPM)
        wheel_rpm = wheel_rps * 60

        return wheel_rpm
        
    def distance_to_line(self, A, B, p):
        # Distance is calculated in meter
        # This method calculates the perpendicular distance from a point p 
        # to a line defined by two points A and B in a 3D space
        num   = np.linalg.norm(np.cross(B - A, A - p)) # calculate cross product 
        denom = np.linalg.norm(B - A) # Euclidean distance between points A and B
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom
    
    # -------------------------------------------------
    # Calculating Euclidean Distance List            |
    # -------------------------------------------------

    def calculate_euc_dist(self, current_waypoint):
        # Define the number of closest waypoints to retrieve
        num_closest_waypoints = 10  # You can adjust this as needed

        # Ensure there are at least 10 waypoints available
        if len(self.route_waypoints) - current_waypoint < num_closest_waypoints:
            raise ValueError("Not enough waypoints available for calculation.")
        
        # Calculate the end index for slicing the route_waypoints list
        end_index = current_waypoint + num_closest_waypoints + 1

        # Retrieve the next 10 waypoints from the current waypoint
        closest_waypoints = self.route_waypoints[current_waypoint + 1:min(end_index, len(self.route_waypoints))]

        # Calculate the Euclidean distance to each waypoint
        euclidean_dist_list = []
        current_waypoint_vector = np.array([self.route_waypoints[current_waypoint].transform.location.x, 
                                        self.route_waypoints[current_waypoint].transform.location.y, 
                                        self.route_waypoints[current_waypoint].transform.location.z])
        
        for waypoint in closest_waypoints:
            # Extract the coordinates from the waypoint object and convert them into a NumPy array
            waypoint_location = np.array([waypoint.transform.location.x, waypoint.transform.location.y, 
                                          waypoint.transform.location.z])
            # Calculate the Euclidean distance from the current_waypoint to the waypoint
            distance = np.linalg.norm(current_waypoint_vector - waypoint_location)
            # Append the distance to the list
            euclidean_dist_list.append(distance)

        return euclidean_dist_list
    
    def vector(self, v):
        # The vector method is a utility function that converts a Carla Location, Vector3D, 
        # or Rotation object to a NumPy array for easier manipulation and calculations.
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])
        
    def calculate_deviation_angle_tan(self, current_waypoint):
        # Define the number of closest waypoints to retrieve
        num_closest_waypoints = 10  # You can adjust this as needed

        # Ensure there are at least 10 waypoints available
        if len(self.route_waypoints) - current_waypoint < num_closest_waypoints:
            raise ValueError("Not enough waypoints available for calculation.")
        
        # Calculate the end index for slicing the route_waypoints list
        end_index = current_waypoint + num_closest_waypoints + 1

        # Get the forward vector of the vehicle
        vehicle_forward_vector = self.vector(self.vehicle.get_transform().rotation.get_forward_vector())[:2]

        # Get the positions of the nearest 10 waypoints
        nearest_waypoints = self.route_waypoints[current_waypoint + 1:min(end_index, len(self.route_waypoints))]

        deviation_angles = []

        for waypoint in nearest_waypoints:
            # Calculate the direction vector from the vehicle to the waypoint
            #waypoint_vector = self.vector(waypoint.transform.location) - self.vector(self.vehicle.get_location())[:2]
            waypoint_vector = self.vector(waypoint.transform.location)[:2] - self.vector(self.vehicle.get_location())[:2]

            # Calculate the angle between the vehicle's forward vector and the direction to the waypoint
            deviation_angle = np.arctan2(waypoint_vector[1], waypoint_vector[0]) - np.arctan2(vehicle_forward_vector[1], vehicle_forward_vector[0])

            # Ensure the angle is within the range [-π, π]
            if deviation_angle > np.pi:
                deviation_angle -= 2 * np.pi
            elif deviation_angle <= -np.pi:
                deviation_angle += 2 * np.pi

            deviation_angles.append(deviation_angle)

        return deviation_angles
    
    # Action space of our vehicle. It can make eight unique actions.
    # Continuous actions are broken into discrete here!
    def angle_diff(self, v0, v1):
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        # If the angular difference is greater than π radians, 
        # it subtracts 2π to bring it within the range [-π, π]:
        if angle > np.pi: angle -= 2 * np.pi
        # If the angular difference is less than or equal to -π radians, 
        # it adds 2π to bring it within the range [-π, π]:
        elif angle <= -np.pi: angle += 2 * np.pi
        return angle
    
    #######################################################
    # For Yaw Acceleration Estimation
    #######################################################

    def calculate_yaw_rate(self, current_yaw):
        if self.previous_yaw is None:
            self.previous_yaw = current_yaw
            return 0  # Yaw rate is 0 for the first measurement

        yaw_rate = (current_yaw - self.previous_yaw) / self.delta_time
        self.previous_yaw = current_yaw  # Update for next iteration
        return yaw_rate

    def calculate_yaw_acceleration(self, current_yaw_rate):
        if self.previous_yaw_rate is None:
            self.previous_yaw_rate = current_yaw_rate
            return 0  # Yaw acceleration is 0 for the first measurement

        yaw_acceleration = (current_yaw_rate - self.previous_yaw_rate) / self.delta_time
        self.previous_yaw_rate = current_yaw_rate  # Update for next iteration
        return yaw_acceleration
    
    #######################################################
    # State Normalization
    #######################################################
    def normalize_state(self, state):
        """Normalize the state using the fitted scaler."""
        state = np.array(state).reshape(1, -1)
        return self.scaler.transform(state)[0]

####################################################### End of SimEnv

#######################################################
# Original Reward Function (JabrahTutorials)
#######################################################
    
def get_reward_comp(vehicle, waypoint, collision):
    vehicle_location = vehicle.get_location()
    x_wp = waypoint.transform.location.x
    y_wp = waypoint.transform.location.y

    x_vh = vehicle_location.x
    y_vh = vehicle_location.y

    wp_array = np.array([x_wp, y_wp])
    vh_array = np.array([x_vh, y_vh])

    dist = np.linalg.norm(wp_array - vh_array)

    vh_yaw = correct_yaw(vehicle.get_transform().rotation.yaw)
    wp_yaw = correct_yaw(waypoint.transform.rotation.yaw)
    cos_yaw_diff = np.cos((vh_yaw - wp_yaw)*np.pi/180.)

    collision = 0 if collision is None else 1
    
    return cos_yaw_diff, dist, collision

def reward_value(cos_yaw_diff, dist, collision, lambda_1=1, lambda_2=1, lambda_3=5):
    reward = (lambda_1 * cos_yaw_diff) - (lambda_2 * dist) - (lambda_3 * collision)
    return reward

#######################################################
# Updated Reward Function as in RL Paper - V1
#######################################################

def calculate_reward_1(theta, d, v, theta_dot):
    """
    Calculate the reward based on the current state.

    :param theta: Current deviation angle from the trajectory
    :param d: Current lateral distance from the trajectory
    :param v: Current velocity of the car
    :param theta_dot: Current yaw acceleration (rate of change of theta)
    :return: The calculated reward
    """
    # Ensure theta and d are within the maximum bounds to avoid negative rewards
    angle_deviation_penalty = max(0, MAX_DEVIATION_ANGLE - abs(theta))
    lateral_distance_penalty = max(0, MAX_DEVIATION_DISTANCE - abs(d))
    # Assuming REWARD_CONSTANT_C is set such that this term is always positive
    comfort_penalty = REWARD_CONSTANT_C - abs(theta_dot)  
    velocity_reward = v / MAX_VELOCITY_THRESHOLD  # Scales with the velocity of the car

    # Compute the reward
    reward = angle_deviation_penalty * lateral_distance_penalty * comfort_penalty * velocity_reward
    return reward

#######################################################
# Updated Reward Function as in RL Paper - V2
#######################################################

def calculate_reward_2(theta, d, v, theta_dot):
    """
    Calculate the reward based on the current state.

    :param theta: Current deviation angle from the trajectory
    :param d: Current lateral distance from the trajectory
    :param v: Current velocity of the car
    :param theta_dot: Current yaw acceleration (rate of change of theta)
    :return: The calculated reward
    """
    # Ensure theta and d are within the maximum bounds to avoid negative rewards
    angle_deviation_penalty = max(0, MAX_DEVIATION_ANGLE - abs(theta))
    lateral_distance_penalty = max(0, MAX_DEVIATION_DISTANCE - abs(d))
    # Assuming REWARD_CONSTANT_C is set such that this term is always positive
    comfort_penalty = REWARD_CONSTANT_C - abs(theta_dot)  
    velocity_reward = v / MAX_VELOCITY_THRESHOLD  # Scales with the velocity of the car

    # Compute the reward
    reward = angle_deviation_penalty * lateral_distance_penalty * comfort_penalty * velocity_reward
    return reward

#######################################################
# Initialize Logger
#######################################################

def initialize_logger(log_file_path):
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Create a file handler for the logger
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    return logger

#######################################################
# Initialize File Writer
#######################################################
def initialize_csv_writer(csv_file_path):
    # Set up CSV writer
    csv_file = open(csv_file_path, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    # Write CSV header
    csv_writer.writerow([
    'Time', 'Episode',
    'EuclideanDist1', 'EuclideanDist2', 'EuclideanDist3', 'EuclideanDist4', 'EuclideanDist5',
    'EuclideanDist6', 'EuclideanDist7', 'EuclideanDist8', 'EuclideanDist9', 'EuclideanDist10',
    'DeviationAngle1', 'DeviationAngle2', 'DeviationAngle3', 'DeviationAngle4', 'DeviationAngle5',
    'DeviationAngle6', 'DeviationAngle7', 'DeviationAngle8', 'DeviationAngle9', 'DeviationAngle10',
    'Velocity', 'VelocityX', 'VelocityY', 'VelocityZ',
    'EngineRPM', 'DistanceFromCenter', 'Angle', 'SteeringCommand', 'Reward', 'TotalReward'
    ])
    return csv_writer, csv_file

#######################################################
# Updated Reward Function as in RL Review Paper - A2
#######################################################

def calculate_reward_A2(current_speed, steering_angle, collision_detected, lane_deviation):
    # Parameters (adjust based on your specific needs)
    # in the A2 paper:
    # velocity      : m/s
    # steering angle: rad
    # lane deviation: m

    speed_limit = 5.0  # Speed limit 5 m/s (18 km/h) assumed by the review paper

    # Constants
    collision_penalty = -10.0  # Penalty for collisions
    out_of_lane_penalty = -1.0  # Penalty for leaving the lane
    constant_step_penalty = -0.1  # Penalty for every step

    # Calculate reward components
    speed_reward = min(current_speed, 10 - current_speed)
    steering_penalty = -0.5 * (steering_angle ** 2)
    collision_reward = collision_penalty if collision_detected else 0
    out_of_lane_reward = out_of_lane_penalty if lane_deviation > 2 else 0

    # Total reward calculation
    reward = speed_reward + steering_penalty + collision_reward + out_of_lane_reward + constant_step_penalty

    return reward