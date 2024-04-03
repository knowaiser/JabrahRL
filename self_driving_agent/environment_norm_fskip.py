import glob
import os
import sys
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt

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
    def __init__(self, visuals=True, target_speed = 18, max_iter = 4000, start_buffer = 10, train_freq = 1,
        action_freq = 4, save_freq = 200, start_ep = 0, max_dist_from_waypoint = 20) -> None:
        

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
        

        # self.spawn_points = self.world.get_map().get_spawn_points()
        # self.spawn_waypoints = self.world.get_map().generate_waypoints(5.0)
        # self.spawn_points = [waypoint.transform for waypoint in self.spawn_waypoints]
        self.spawn_points = self.generate_custom_spawn_points(distance = 5.0)

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
        self.closest_waypoint_index = None
        self.route_waypoints = []
        
    
    def reset(self):
        for actor in self.actor_list:
            if actor.is_alive: # this line is newly added
                actor.destroy()
           
    def end_episode(self):
        # If you have any actors (vehicles, sensors, etc.) that need to be destroyed
        for actor in self.actor_list:
            if actor.is_alive:
                actor.destroy()
        self.actor_list.clear()  # Clear the list of actors for the next episode

        # Reset environment-specific variables
        self.route_waypoints = []  # Assuming this is where waypoints are stored
        self.current_waypoint_index = 0  # Reset index or similar variables
        # self.total_reward = 0  # If you're tracking rewards

        # Reset any other state or variables related to the episode
        # For example, if you're tracking episode length or time
        self.episode_length = 0

        # Add any additional cleanup or state resetting you require here
        # This is also where you might reset the simulation environment if needed
        if self.world is not None:
            self.reset()  # Assuming your simulation environment has a reset method

        # Log the episode's end if necessary
        print(f"Episode ended. Preparing for a new episode...")

   
    def generate_episode(self, ep, eval=True):
        with CarlaSyncMode(self.world, self.camera_rgb, self.camera_rgb_vis, self.collision_sensor, fps=30) as sync_mode:
            counter = 0
            episode_reward = 0

            # Generate route UPDATED 
            self.generate_route_2()


            # TO DRAW WAYPOINTS
            for w in self.route_waypoints:
                self.world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
                                            color=carla.Color(r=0, g=0, b=255), life_time=120.0,
                                            persistent_lines=True)
                
            for i, waypoint in enumerate(self.route_waypoints[:5]):  # Just as an example, adjust the range as needed
                print(f"Waypoint {i}: Location = {waypoint.transform.location}")
            
            # for i, waypoint in enumerate(self.route_waypoints):
            #     next_waypoints = waypoint.next(5.0)  # This gets the next waypoints within 5 meters
            #     print(f"Waypoint {i} has {len(next_waypoints)} next waypoints.")

            # # Save graph of plotted points as route.png
            # x_graph = [p.transform.location.x for p in self.route_waypoints]
            # y_graph = [p.transform.location.y for p in self.route_waypoints]
            # plt.clf()  # Clear the figure to ensure no old data is plotted
            # plt.plot(x_graph, y_graph, marker = 'o')
            # # plt.savefig(f"route_ep_{ep}.png")
            # plt.savefig(f"route_ep_{ep}.jpg", quality = 45) # MAX qulaity = 95

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


            try:

                # Capture the initial state
                next_state = self.capture_states()

                if next_state is None:
                    print("Preparing for a new episode...")
                    self.reset()  # Clean up and log the end of the episode, if needed
                    return None  # Break out of the loop to finish the current episode

                while True: # simulation loop
                    if self.visuals: # Check the visuals and maintain a consistent frame rate
                        if should_quit(): # utils.py, checks pygame for quit events
                            return
                        self.clock.tick_busy_loop(30) # does not advance the simulation, only controls the fps rate

                    # Advance simulation
                    state = next_state
                    counter +=1
                    self.global_t +=1


                    # Apply model to get steering angle
                    action = self.agent.choose_action(state)
                    steer_nd = action
                    steer = steer_nd.item()

                    control = self.speed_controller.run_step(self.target_speed)
                    control.steer = steer
                    self.vehicle.apply_control(control)

                    fps = round(1.0 / snapshot.timestamp.delta_seconds)

                    # NEXT TICK
                    returned_data = sync_mode.tick(timeout=2.0)
                    # Assuming the first item in returned_data is the snapshot
                    snapshot = returned_data[0]
                    # Assuming the last item is the collision data
                    collision = returned_data[-1]
                    # Retrieving image_rgb_vis, which should be the second-to-last item
                    image_rgb_vis = returned_data[-2]

                    # Capture the next state
                    next_state = self.capture_states()
                    
                    if next_state is None:
                        print("Preparing for a new episode...")
                        self.reset()  # Clean up and log the end of the episode, if needed
                        return None  # Break out of the loop to finish the current episode
                    
                    velocity = self.vehicle.get_velocity()

                    # Print the closest waypoint index
                    print("Episode waypoint index:", self.closest_waypoint_index)

                    # OLD REWARD CALL
                    cos_yaw_diff, dist, collision = get_reward_comp(self.vehicle, 
                                                    self.route_waypoints[self.closest_waypoint_index], collision)
                    # reward = reward_value(cos_yaw_diff, dist, collision)

                    # UPDATED REWARD
                    # reward = calculate_reward_1(self.angle, self.distance_from_center, self.velocity, yaw_acceleration)
                    
                    reward = calculate_reward_A2(velocity.x, steer, collision, self.angle)

                    #if snapshot is None or image_rgb is None:
                    if snapshot is None:
                        print("Process ended here")
                        break


                    done = 1 if collision else 0

                    episode_reward += reward
                    self.total_rewards += reward

                    # Logging state before normalization
                    # self.logger.info(f"Pre-Norm State: {state}")
                    self.csv_writer.writerow([time.time(), ep, self.closest_waypoint_index,
                                            *state, steer, reward, episode_reward, self.total_rewards])
                    self.csv_writer_norm.writerow([time.time(), ep, self.closest_waypoint_index,
                                            *next_state, steer, reward, episode_reward, self.total_rewards])

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
                    # if self.scaler_fitted:
                    #     state = self.normalize_state(state)
                    #     self.logger.info(f"Post-Norm State: {state}")
                    #     # self.csv_writer_norm.writerow([time.time(), ep, *self.euclidean_dist_list, *self.dev_angle_array,
                    #     #   self.velocity, self.velocity_x, self.velocity_y, self.velocity_z,
                    #     #   self.engine_rpm, self.distance_from_center, self.angle])
                    #     self.csv_writer_norm.writerow([time.time(), ep, *state, steer, reward, self.total_rewards])



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

                    # if collision == 1 or counter >= self.max_iter or dist > self.max_dist_from_waypoint:
                    #     print("Episode {} processed".format(ep), counter)
                    #     break
                    
                    if collision == 1:
                        print(f"Episode {ep} ended due to a collision after {counter} iterations.")
                        break
                    elif counter >= self.max_iter:
                        print(f"Episode {ep} reached the maximum iteration limit of {self.max_iter}.")
                        break
                    elif dist > self.max_dist_from_waypoint:
                        print(f"Episode {ep} ended because the vehicle was more than {self.max_dist_from_waypoint} meters from the closest waypoint.")
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

    def calculate_euc_dist(self, current_waypoint_index):
        ##########################################################
        # This version raises an error and stops the simulation when there are not enought waypoints
        ##########################################################
        # Define the number of closest waypoints to retrieve
        num_closest_waypoints = 10  # You can adjust this as needed

        # Ensure there are at least 10 waypoints available
        if len(self.route_waypoints) - current_waypoint_index < num_closest_waypoints:
            raise ValueError("Not enough waypoints available for calculation.")
        
        # Calculate the end index for slicing the route_waypoints list
        end_index = current_waypoint_index + num_closest_waypoints + 1

        # Retrieve the next 10 waypoints from the current waypoint
        closest_waypoints = self.route_waypoints[current_waypoint_index + 1:min(end_index, len(self.route_waypoints))]

        # Calculate the Euclidean distance to each waypoint
        euclidean_dist_list = []
        current_waypoint_vector = np.array([self.route_waypoints[current_waypoint_index].transform.location.x, 
                                        self.route_waypoints[current_waypoint_index].transform.location.y, 
                                        self.route_waypoints[current_waypoint_index].transform.location.z])
        
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
        
    def calculate_deviation_angle_tan(self, current_waypoint_index):
        ##########################################################
        # This version raises an error and stops the simulation when there are not enought waypoints
        ##########################################################
        # Define the number of closest waypoints to retrieve
        num_closest_waypoints = 10  # You can adjust this as needed

        # Ensure there are at least 10 waypoints available
        if len(self.route_waypoints) - current_waypoint_index < num_closest_waypoints:
            raise ValueError("Not enough waypoints available for calculation.")
        
        # Calculate the end index for slicing the route_waypoints list
        end_index = current_waypoint_index + num_closest_waypoints + 1

        # Get the forward vector of the vehicle
        vehicle_forward_vector = self.vector(self.vehicle.get_transform().rotation.get_forward_vector())[:2]

        # Get the positions of the nearest 10 waypoints
        nearest_waypoints = self.route_waypoints[current_waypoint_index + 1:min(end_index, len(self.route_waypoints))]

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
    
    #######################################################
    # Capture States
    #######################################################
    def capture_states(self):

        # Location of the car
        self.location = self.vehicle.get_location()
        #waypoint = self.world.get_map().get_waypoint(self.location, project_to_road=True, 
        #    lane_type=carla.LaneType.Driving)

        # Determine the current waypoint index
        # closest_waypoint_index = self.determine_current_waypoint_index()
        next_wp, next_next_wp, next_wp_index, next_next_wp_index = self.get_next_two_waypoints_and_indices()

        # CHECK if next_wp or next_next_wp is None, terminate episode
        if next_wp is None or next_next_wp is None:
            print("Next waypoint(s) not found. Ending episode.")
            self.reset()  # Gracefully end the episode and prepare for a new one
            return

        self.closest_waypoint_index = next_wp_index

        # Print waypoints and thier indices:
        if next_wp is not None:
            print(f"Next waypoint index: {next_wp_index}, Location: {next_wp.transform.location}")
        else:
            print("Next waypoint not found.")

        if next_next_wp is not None:
            print(f"Next next waypoint index: {next_next_wp_index}, Location: {next_next_wp.transform.location}")
        else:
            print("Next next waypoint not found.")

        # Draw next waypoint
        # self.world.debug.draw_point(next_wp.transform.location, life_time=5)

        
        # Retrieve vehicle's current location and velocity
        velocity = self.vehicle.get_velocity()
        self.velocity_x = velocity.x * 3.6
        self.velocity_y = velocity.y * 3.6
        self.velocity_z = velocity.z * 3.6
        self.velocity = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6

        # Estimate engine and wheel RPM
        self.engine_rpm = self.estimate_engine_rpm()
        self.wheel_rpm = self.estimate_wheel_rpm(velocity)

        # Retrieve current yaw and calculate yaw rate and acceleration
        current_yaw = self.vehicle.get_transform().rotation.yaw
        current_yaw_rate = self.calculate_yaw_rate(current_yaw)
        yaw_acceleration = self.calculate_yaw_acceleration(current_yaw_rate)

        # self.closest_waypoint_index = closest_waypoint_index
        # current_waypoint = self.route_waypoints[closest_waypoint_index]
        # next_waypoint = self.route_waypoints[closest_waypoint_index + 1]

        # CALCULATE d (distance_from_center) and Theta (angle)
        # The result is the distance of the vehicle from the center of the lane:
        self.distance_from_center = self.distance_to_line(self.vector(next_wp.transform.location),
                                                          self.vector(next_next_wp.transform.location),
                                                          self.vector(self.location))
        # Get angle difference between closest waypoint and vehicle forward vector
        fwd    = self.vector(self.vehicle.get_velocity())
        wp_fwd = self.vector(next_wp.transform.rotation.get_forward_vector()) # Return: carla.Vector3D
        self.angle  = self.angle_diff(fwd, wp_fwd)

        # Update Euclidean distances and deviation angles lists
        self.euclidean_dist_list = self.calculate_euc_dist(next_wp_index)
        self.dev_angle_array = self.calculate_deviation_angle_tan(next_wp_index)

        # Package the state information
        state = np.array([*self.euclidean_dist_list, *self.dev_angle_array,
                        self.velocity, self.velocity_x, self.velocity_y, self.velocity_z,
                        self.engine_rpm, self.distance_from_center, self.angle])

        return state
    
    
    #######################################################
    # Generate Route
    #######################################################
    
    def generate_route(self, total_distance=780):
        """
        Generates a route based on the vehicle's current location and a specified total distance.

        Args:
        - total_distance: The total distance of the route to generate.
        """
        # Initialize the route waypoints list and current waypoint index
        self.route_waypoints = []
        self.current_waypoint_index = 0
        self.total_distance = total_distance  # You can set this depending on the town

        # Get the initial waypoint based on the vehicle's current location
        current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location(),
                                                             project_to_road=True,
                                                             lane_type=carla.LaneType.Driving)
        self.route_waypoints.append(current_waypoint)

        # Generate the rest of the waypoints for the route
        for x in range(self.total_distance):
            # Depending on the section of the route, select the appropriate next waypoint
            if x < 650:
                next_waypoint = current_waypoint.next(10.0)[0]
            else:
                next_waypoint = current_waypoint.next(10.0)[-1]

            self.route_waypoints.append(next_waypoint)
            current_waypoint = next_waypoint


    #######################################################
    # Generate Route - source: https://medium.com/@chardorn/creating-carla-waypoints-9d2cc5c6a656
    #######################################################
    
    def generate_route_2(self):
        """
        Generates waypoints based on the vehicle's lane 

        Positive lane_id values represent lanes to the right of the centerline of the road 
        (when facing in the direction of increasing waypoint s-values, 
        which usually corresponds to the driving direction).
        
        Negative lane_id values represent lanes to the left of the centerline of the road.
        
        The magnitude of the lane_id increases as you move further from the road's centerline,
        meaning lane_id = 1 or lane_id = -1 indicates the lane immediately adjacent to the centerline of the road.
        """
        # Initialize route_waypoints list
        self.route_waypoints = []

        waypoint_list = self.world.get_map().generate_waypoints(5.0)
        print("Length: " + str(len(waypoint_list)))

        # Determine the vehicle's lane
        vehicle_location = self.vehicle.get_location()

        # Get the waypoint corresponding to the vehicle's location
        waypoint = self.world.get_map().get_waypoint(vehicle_location)

        # Retrieve the lane ID
        target_lane = waypoint.lane_id

        # Retrieve the waypoints from that lane and make them the vehicle's route
        for i in range(len(waypoint_list) - 1):
            if waypoint_list[i].lane_id == target_lane:
                self.route_waypoints.append(waypoint_list[i])

    
    #######################################################
    # Determine the current waypoint index
    #######################################################
    
    def determine_current_waypoint_index(self):
        closest_waypoint_index = None
        max_dot_product = -float('inf')  # Initialize with very small number
        vehicle_location = self.vehicle.get_location()
        vehicle_forward_vector = self.vehicle.get_transform().get_forward_vector()
        
        for i, waypoint in enumerate(self.route_waypoints):
            waypoint_vector = waypoint.transform.location - vehicle_location
            dot_product = vehicle_forward_vector.dot(waypoint_vector)
            
            if dot_product > max_dot_product:
                max_dot_product = dot_product
                closest_waypoint_index = i
                
        return closest_waypoint_index
    
    #######################################################
    # Get next waypoint
    #######################################################
    
    def get_next_waypoint(self):
        vehicle_location = self.vehicle.get_location()
        min_distance = 1000
        next_waypoint = None

        for waypoint in self.route_waypoints:
            waypoint_location = waypoint.transform.location

            # Only check waypoints that are in the front of the vehicle 
            # (if x is negative, then the waypoint is to the rear)
            #TODO: Check if this applies for all maps
            if (waypoint_location - vehicle_location).x > 0:

                # Find the waypoint closest to the vehicle, 
                # but once vehicle is close to upcoming waypoint, search for next one
                if vehicle_location.distance(waypoint_location) < min_distance and vehicle_location.distance(waypoint_location) > 5:
                    min_distance = vehicle_location.distance(waypoint_location)
                    next_waypoint = waypoint

        return next_waypoint
    
    #######################################################
    # Get next two waypoints and their indices
    #######################################################
    
    def get_next_two_waypoints_and_indices(self):
        vehicle_location = self.vehicle.get_location()
        min_distance = 1000
        next_waypoint = None
        next_next_waypoint = None
        index_of_next_waypoint = None
        index_of_next_next_waypoint = None

        if len(self.route_waypoints) > 0:
            print(f"First waypoint location: {self.route_waypoints[0].transform.location}")
            print(f"Last waypoint location: {self.route_waypoints[-1].transform.location}")

        print(f"Current vehicle location: {vehicle_location}")

        # Find the closest waypoint ahead of the vehicle
        for index, waypoint in enumerate(self.route_waypoints):
            waypoint_location = waypoint.transform.location

            # Assuming the vehicle's forward direction corresponds with increasing waypoint index
            distance = vehicle_location.distance(waypoint_location)
            if distance < min_distance and (waypoint_location - vehicle_location).x > 0:
                min_distance = distance
                next_waypoint = waypoint
                index_of_next_waypoint = index

        # If a closest waypoint is found, attempt to get the next waypoint in the list
        if next_waypoint is not None and index_of_next_waypoint is not None:
            try:
                next_next_waypoint = self.route_waypoints[index_of_next_waypoint + 1]
                index_of_next_next_waypoint = index_of_next_waypoint + 1
            except IndexError:
                next_next_waypoint = None  # This might happen if the next waypoint is the last one
                index_of_next_next_waypoint = None

        return next_waypoint, next_next_waypoint, index_of_next_waypoint, index_of_next_next_waypoint
    
    def generate_custom_spawn_points(self, distance=5.0):
        map = self.world.get_map()
        # Generate waypoints across the map at specified intervals
        waypoints = map.generate_waypoints(distance)
        custom_spawn_points = []

        for waypoint in waypoints:
            # Optionally filter waypoints; for example, ensure they are in driving lanes
            if waypoint.lane_type == carla.LaneType.Driving:
                # Create a Transform with the waypoint's location and rotation
                location = waypoint.transform.location
                # Adjust Z location to ensure the vehicle spawns above the ground
                location.z += 2
                rotation = waypoint.transform.rotation
                spawn_transform = carla.Transform(location, rotation)

                custom_spawn_points.append(spawn_transform)

        return custom_spawn_points


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
    'Time', 'Episode', 'ClosestWaypointIndex'
    'EuclideanDist1', 'EuclideanDist2', 'EuclideanDist3', 'EuclideanDist4', 'EuclideanDist5',
    'EuclideanDist6', 'EuclideanDist7', 'EuclideanDist8', 'EuclideanDist9', 'EuclideanDist10',
    'DeviationAngle1', 'DeviationAngle2', 'DeviationAngle3', 'DeviationAngle4', 'DeviationAngle5',
    'DeviationAngle6', 'DeviationAngle7', 'DeviationAngle8', 'DeviationAngle9', 'DeviationAngle10',
    'Velocity', 'VelocityX', 'VelocityY', 'VelocityZ',
    'EngineRPM', 'DistanceFromCenter', 'Angle', 'SteeringCommand', 'Reward', 'EpisodeReward', 'TotalReward'
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