import time
import random
import numpy as np
import pygame
import sklearn
from simulation.connection import carla
from simulation.sensors import CameraSensor, CameraSensorEnv, CollisionSensor
from simulation.settings import *
from can_interface import CAN
from settings import *
from sklearn.preprocessing import MinMaxScaler


class CarlaEnvironment():

    def __init__(self, client, world, town, checkpoint_frequency=100, continuous_action=True) -> None:


        self.client = client
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.action_space = self.get_discrete_action_space()
        self.continous_action_space = continuous_action
        self.display_on = VISUAL_DISPLAY
        self.vehicle = None
        self.settings = None
        self.current_waypoint_index = 0
        self.checkpoint_waypoint_index = 0
        self.fresh_start=True
        self.checkpoint_frequency = checkpoint_frequency
        self.route_waypoints = None
        self.town = town
        
        # Objects to be kept alive
        self.camera_obj = None
        self.env_camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None

        # Two very important lists for keeping track of our actors and their observations.
        self.sensor_list = list()
        self.actor_list = list()
        self.walker_list = list()
        self.create_pedestrians()

        # Create an instance of the CAN class
        # self.can_instance = CAN()



    # A reset function for resetting our environment.
    def reset(self):

        try:
            
            if len(self.actor_list) != 0 or len(self.sensor_list) != 0:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                self.sensor_list.clear()
                self.actor_list.clear()
            self.remove_sensors()


            # Blueprint of our main vehicle
            vehicle_bp = self.get_vehicle(CAR_NAME)

            # the total distance for the planned trajectory (self.total_distance)
            # is predefined based on the town setting
            # self.map.get_spawn_points returns list(carla.Transform)
            if self.town == "Town07":
                transform = self.map.get_spawn_points()[38] #Town7  is 38 
                self.total_distance = 750
            elif self.town == "Town02":
                transform = self.map.get_spawn_points()[1] #Town2 is 1
                self.total_distance = 780
            else:
                transform = random.choice(self.map.get_spawn_points())
                self.total_distance = 250

            self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform) # Return: carla.Actor
            self.actor_list.append(self.vehicle)


            # Camera Sensor
            self.camera_obj = CameraSensor(self.vehicle) # a custom class in sensors.py
            while(len(self.camera_obj.front_camera) == 0):
                time.sleep(0.0001)
            self.image_obs = self.camera_obj.front_camera.pop(-1)
            self.sensor_list.append(self.camera_obj.sensor)

            # Third person view of our vehicle in the Simulated env
            if self.display_on:
                self.env_camera_obj = CameraSensorEnv(self.vehicle) # a custom class in sensors.py
                self.sensor_list.append(self.env_camera_obj.sensor)

            # Collision sensor
            self.collision_obj = CollisionSensor(self.vehicle) # a custom class in sensors.py
            self.collision_history = self.collision_obj.collision_data
            self.sensor_list.append(self.collision_obj.sensor)

            #
            self.timesteps = 0
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.previous_location = self.vehicle.get_location()
            self.distance_traveled = 0.0
            self.center_lane_deviation = 0.0
            self.target_speed = 22 #km/h
            self.max_speed = 25.0
            self.min_speed = 15.0
            self.max_distance_from_center = 3
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
            # add states here K
            # STATES
            # 1- Euclidean distances and angle deviations to 10 waypoints: euclidean_dist_list, dev_angle_list
            # The waypoints are spaced 5 meters apart, providing a detailed trajectory for the vehicle to follow.
            # 2- The car's velocities in three dimensions (x, y, z): velocity (3D), velocity_x, velocity_y, velocity_z
            # 3- Engine RPM (revolutions per minute). can be calculated: engine_rpm
            # 4- Wheel speeds: can be retrieved from CAN bus data
            # 5- Trajectory deviation distance and angle: distance_from_center, angle 
            # 6- Road curvatures. can be calculated
            
            # PERFORMANCE METRICS (comparison between RL and PID)
            # 1- Yaw Acceleration (rad/s^2): can be calculated from yaw angle and yaw rate
            # 2- Steering Angle (rad): action
            # 3- Angle Deviation (rad): deviation_avg
            # 4- Distance Deviation (m): center_lane_deviation

            # Number of states = (10 + 10) + (4) + (1) + (2) = 27


            self.euclidean_dist_list = []
            self.dev_angle_list = []
            self.dev_dist = float(0.0)
            self.engine_rpm = float(0.0)
            self.deviation_avg = float(0.0)
            # self.wheel_speeds = []
            


            if self.fresh_start:
                self.current_waypoint_index = 0
                # Waypoint nearby angle and distance from it
                self.route_waypoints = list() # initializes an empty list named route_waypoints
                
                # the line fetches the waypoint on the driving lane of the road network 
                # that corresponds to the current location of the vehicle:
                self.waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
                
                # current_waypoint is a temporary variable
                # represents the current waypoint in the loop:
                current_waypoint = self.waypoint
                
                # self.route_waypoints is a list that will store the sequence of waypoints
                # representing the planned trajectory or route:
                self.route_waypoints.append(current_waypoint)
                
                # This loop is responsible for iteratively determining the next waypoints 
                # along the planned trajectory and adding them to the self.route_waypoints list: 
                for x in range(self.total_distance):
                    if self.town == "Town07":
                        if x < 650:
                            # current_waypoint.next(5.0): Retrieves the next waypoints on the lane, 
                            # considering a distance of 1.0 meter ahead from the current waypoint.
                            # [0]: Accesses the first element in the list of waypoints returned by current_waypoint.next(5.0). 
                            # This is done to select a specific waypoint among the available options.
                            next_waypoint = current_waypoint.next(5.0)[0]
                        else:
                            next_waypoint = current_waypoint.next(5.0)[-1]
                    elif self.town == "Town02":
                        if x < 650:
                            next_waypoint = current_waypoint.next(5.0)[-1]
                        else:
                            next_waypoint = current_waypoint.next(5.0)[0]
                    else:
                        next_waypoint = current_waypoint.next(5.0)[0]
                    self.route_waypoints.append(next_waypoint)
                    current_waypoint = next_waypoint
            else:
                # Teleport vehicle to last checkpoint
                # This block of code ensures that when the environment is not at a fresh start,
                # the vehicle is teleported to the last checkpoint to resume the trajectory from that point
                waypoint = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
                transform = waypoint.transform
                self.vehicle.set_transform(transform)
                self.current_waypoint_index = self.checkpoint_waypoint_index

            # initial observations
            # Convert lists to NumPy arrays
            euclidean_dist_array = np.array(self.euclidean_dist_list)
            dev_angle_array = np.array(self.dev_angle_list)
            # Create the navigation_obs NumPy array
            self.navigation_obs = np.array([self.throttle, self.velocity, self.velocity_x, self.velocity_y,self.velocity_z,
                                            self.previous_steer, self.distance_from_center, self.angle, self.euclidean_dist_array,
                                            self.dev_angle_array, self.dev_dist, self.engine_rpm, self.deviation_avg])

                        
            time.sleep(0.5)
            self.collision_history.clear()

            self.episode_start_time = time.time()
            return [self.image_obs, self.navigation_obs]
            #return [self.navigation_obs] # we don't need images in our implementation

        except:
            # apply_batch(self, commands):
            # Executes a list of commands on a single simulation step and retrieves no information
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()


# ----------------------------------------------------------------
# Step method is used for implementing actions taken by our agent|
# ----------------------------------------------------------------

    # A step function is used for taking inputs generated by neural net.
    def step(self, action_idx):
        try:
            # incrementing the time step counter 
            # and setting the fresh_start flag to False
            self.timesteps+=1
            self.fresh_start = False

            # self.navigation_obs = np.array([self.throttle, self.velocity, self.velocity_x, self.velocity_y,self.velocity_z,
            #                               self.previous_steer, self.distance_from_center, self.angle, self.euclidean_dist_array,
            #                               self.dev_angle_array, self.dev_dist, self.engine_rpm, self.deviation_avg])

            # Retrieve the velocity of the vehicle
            # and calculate it in km/h
            velocity = self.vehicle.get_velocity() # Return: carla.Vector3D - m/s
            self.velocity_x = velocity.x * 3.6
            self.velocity_y = velocity.y * 3.6
            self.velocity_z = velocity.z * 3.6
            self.velocity = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6

            # Retrieve the wheel speed values from the CAN object
            # self.wheel_speeds = self.can_bus.read_wheel_speeds()
            
            # Action from action space for controlling the vehicle with a discrete action
            if self.continous_action_space:
                # action_idx is assumed to be a tuple or list 
                # representing the agent's chosen action:
                steer = float(action_idx[0])
                # steer is clamped between -1.0 and 1.0 
                # to ensure it falls within a valid range for steering:
                steer = max(min(steer, 1.0), -1.0)

                # This ensures that the throttle value is within the range [0, 1.0]: 
                throttle = float((action_idx[1] + 1.0)/2)
                throttle = max(min(throttle, 1.0), 0.0)

                # This blending of values with weights (0.9 and 0.1) is likely done 
                # to provide a smoother transition between consecutive control inputs:
                # throttle [0.0, 1.0]. Default is 0.0
                # steering [-1.0, 1.0]. Default is 0.0

                # Applying smoothed control inputs to the vehicle
                self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1, throttle=self.throttle*0.9 + throttle*0.1))
                self.previous_steer = steer
                self.throttle = throttle
            else:
                steer = self.action_space[action_idx]
                if self.velocity < 20.0:
                    self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1, throttle=1.0))
                else:
                    self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1))
                self.previous_steer = steer
                # self.throttle is a constant value of 1.0 when the velocity is greater than or equal to 20.0
                self.throttle = 1.0
            
            # Traffic Light state
            if self.vehicle.is_at_traffic_light():
                traffic_light = self.vehicle.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_light.set_state(carla.TrafficLightState.Green)

            self.collision_history = self.collision_obj.collision_data            

            # Estimate the engine_rpm
            self.engine_rpm = self.estimate_engine_rpm()

            # Rotation of the vehicle in correlation to the map/lane
            # get_transform(self): Returns the actor's transform (location and rotation)
            # the client received during last tick. The method does not call the simulator
            # rotation (carla.Rotation - degrees (pitch, yaw, roll))
            self.rotation = self.vehicle.get_transform().rotation.yaw 
            # Location of the car
            self.location = self.vehicle.get_location()

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
            self.current_waypoint = self.route_waypoints[ self.current_waypoint_index    % len(self.route_waypoints)]
            self.next_waypoint = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
            # The result is the distance of the vehicle from the center of the lane:
            self.distance_from_center = self.distance_to_line(self.vector(self.current_waypoint.transform.location),self.vector(self.next_waypoint.transform.location),self.vector(self.location))
            # Calculate calculate_euc_dist
            self.euclidean_dist_list = self.calculate_euc_dist(self.vector(self.current_waypoint.transform.location))
            # self.center_lane_deviation: This variable is a running sum that accumulates the deviation
            # of the vehicle from the center of the lane
            # self.distance_from_center: This is the current deviation calculated in the previous line
            self.center_lane_deviation += self.distance_from_center

            # Calculate dev_angle_list
            self.dev_angle_array = self.calculate_deviation_angle_tan()

            # Get angle difference between closest waypoint and vehicle forward vector
            fwd    = self.vector(self.vehicle.get_velocity())
            wp_fwd = self.vector(self.current_waypoint.transform.rotation.get_forward_vector()) # Return: carla.Vector3D
            self.angle  = self.angle_diff(fwd, wp_fwd)

             # Update checkpoint for training
            if not self.fresh_start:
                if self.checkpoint_frequency is not None:
                    # // is integer division to determine how many times 
                    # the current waypoint index has passed the checkpoint frequency
                    self.checkpoint_waypoint_index = (self.current_waypoint_index // self.checkpoint_frequency) * self.checkpoint_frequency

            
            # Rewards are given below!
            done = False
            reward = 0

            # Checks if there have been collisions during the episode
            if len(self.collision_history) != 0:
                done = True
                reward = -10
            # Checks if the vehicle deviates too far from the center of the lane
            elif self.distance_from_center > self.max_distance_from_center:
                done = True
                reward = -10
            # Checks if 10 seconds have passed since the episode started, 
            # and the velocity of the vehicle is less than 1.0
            elif self.episode_start_time + 10 < time.time() and self.velocity < 1.0:
                reward = -10
                done = True
            # Checks if the velocity of the vehicle exceeds the maximum allowed speed
            elif self.velocity > self.max_speed:
                reward = -10
                done = True

            # Interpolated from 1 when centered to 0 when 3 m from center
            # centering_factor: represents how well the vehicle is centered within the lane:
            #centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)
            # Interpolated from 1 when aligned with the road to 0 when +/- 30 degrees of road
            # angle_factor: represents how well the vehicle is aligned with the road direction:
            #angle_factor = max(1.0 - abs(self.angle / np.deg2rad(20)), 0.0)

            # These lines calculate the reward for the reinforcement learning agent 
            # based on its behavior and the environmental factors
            if not done:
                #if self.continous_action_space:
                    # If the velocity is below the minimum speed (self.velocity < self.min_speed), 
                    # the reward is scaled based on the ratio of the current velocity to the minimum speed
                    #if self.velocity < self.min_speed:
                        #reward = (self.velocity / self.min_speed) * centering_factor * angle_factor    
                    # If the velocity exceeds the target speed (self.velocity > self.target_speed), 
                    # the reward is scaled based on the ratio of the difference between 
                    # the current velocity and the target speed to the speed range (max_speed - target_speed).
                    #elif self.velocity > self.target_speed:               
                        #reward = (1.0 - (self.velocity-self.target_speed) / (self.max_speed-self.target_speed)) * centering_factor * angle_factor  
                    # otherwise, reward = 1
                    #else:                                         
                        #reward = 1.0 * centering_factor * angle_factor 
                #else: # discrete action space
                    #reward = 1.0 * centering_factor * angle_factor
                # IMPLEMENTING CUSTOM REWARD FUNCTION
                reward = self.reward_function(self.euclidean_dist_list[0], self.dev_angle_array[0], self.velocity)

            # the logic to determine if the episode should terminate
            if self.timesteps >= 7500:
                done = True
            # This condition is likely related to reaching the end of the planned trajectory:
            elif self.current_waypoint_index >= len(self.route_waypoints) - 2:
                done = True
                self.fresh_start = True
                if self.checkpoint_frequency is not None:
                    if self.checkpoint_frequency < self.total_distance//2:
                        self.checkpoint_frequency += 2
                    else:
                        self.checkpoint_frequency = None
                        self.checkpoint_waypoint_index = 0

            # The purpose of this loop is likely to wait until an image becomes available in the front_camera list. 
            # It's a way of synchronizing the code and ensuring that 
            # an image is ready to be processed before proceeding further. 
            # Once the loop exits, it implies that there is at least one image in the front_camera list:
            while(len(self.camera_obj.front_camera) == 0):
                time.sleep(0.0001)

            # pop(-1) removes and returns the last element from the list
            # retrieves the latest image from the camera
            self.image_obs = self.camera_obj.front_camera.pop(-1)
            normalized_velocity = self.velocity/self.target_speed
            normalized_distance_from_center = self.distance_from_center / self.max_distance_from_center
            normalized_angle = abs(self.angle / np.deg2rad(20))
            self.navigation_obs = np.array([self.throttle, self.velocity, normalized_velocity, normalized_distance_from_center, normalized_angle])
            # two parts of observations is set above: image_obs and navigation_obs

            # Remove everything that has been spawned in the env
            # cleanup process after an episode is done or terminated:
            if done:
                # a measure of average deviation per timestep
                self.center_lane_deviation = self.center_lane_deviation / self.timesteps
                # a measure of the distance covered during the episode
                self.distance_covered = abs(self.current_waypoint_index - self.checkpoint_waypoint_index)
                
                for sensor in self.sensor_list:
                    sensor.destroy()
                
                self.remove_sensors()
                
                for actor in self.actor_list:
                    actor.destroy()
            
            return [self.image_obs, self.navigation_obs], reward, done, [self.distance_covered, self.center_lane_deviation]

        except:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()

        

# -------------------------------------------------
# Creating and Spawning Pedestrians in our world |
# -------------------------------------------------

    # Walkers are to be included in the simulation yet!
    def create_pedestrians(self):
        try:

            # Our code for this method has been broken into 3 sections.

            # 1. Getting the available spawn points in  our world.
            # Random Spawn locations for the walker
            walker_spawn_points = []
            for i in range(NUMBER_OF_PEDESTRIAN):
                spawn_point_ = carla.Transform()
                loc = self.world.get_random_location_from_navigation() # Return: carla.Location
                if (loc != None):
                    spawn_point_.location = loc
                    walker_spawn_points.append(spawn_point_)

            # 2. We spawn the walker actor and ai controller
            # Also set their respective attributes
            for spawn_point_ in walker_spawn_points:
                walker_bp = random.choice(
                    self.blueprint_library.filter('walker.pedestrian.*'))
                walker_controller_bp = self.blueprint_library.find(
                    'controller.ai.walker')
                # Walkers are made visible in the simulation
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                # They're all walking not running on their recommended speed
                if walker_bp.has_attribute('speed'):
                    walker_bp.set_attribute(
                        'speed', (walker_bp.get_attribute('speed').recommended_values[1]))
                else:
                    walker_bp.set_attribute('speed', 0.0)
                walker = self.world.try_spawn_actor(walker_bp, spawn_point_)
                if walker is not None:
                    walker_controller = self.world.spawn_actor(
                        walker_controller_bp, carla.Transform(), walker)
                    self.walker_list.append(walker_controller.id)
                    self.walker_list.append(walker.id)
            all_actors = self.world.get_actors(self.walker_list)

            # set how many pedestrians can cross the road
            #self.world.set_pedestrians_cross_factor(0.0)
            # 3. Starting the motion of our pedestrians
            for i in range(0, len(self.walker_list), 2):
                # start walker
                all_actors[i].start() # start(self): Enables AI control for its parent walker
            # set walk to random point
                all_actors[i].go_to_location(
                    self.world.get_random_location_from_navigation())

        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.walker_list])


# ---------------------------------------------------
# Creating and Spawning other vehciles in our world|
# ---------------------------------------------------


    def set_other_vehicles(self):
        try:
            # NPC vehicles generated and set to autopilot
            # One simple for loop for creating x number of vehicles and spawning them into the world
            for _ in range(0, NUMBER_OF_VEHICLES):
                # Choose a random spawn point from the available spawn points in the map
                spawn_point = random.choice(self.map.get_spawn_points())
                # Choose a random blueprint for a vehicle from the available vehicle blueprints 
                bp_vehicle = random.choice(self.blueprint_library.filter('vehicle'))
                other_vehicle = self.world.try_spawn_actor(
                    bp_vehicle, spawn_point)
                if other_vehicle is not None:
                    other_vehicle.set_autopilot(True)
                    self.actor_list.append(other_vehicle)
            print("NPC vehicles have been generated in autopilot mode.")
        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.actor_list])


# ----------------------------------------------------------------
# Extra very important methods: their names explain their purpose|
# ----------------------------------------------------------------

    # Setter for changing the town on the server.
    def change_town(self, new_town):
        self.world = self.client.load_world(new_town)


    # Getter for fetching the current state of the world that simulator is in.
    def get_world(self) -> object:
        return self.world


    # Getter for fetching blueprint library of the simulator.
    def get_blueprint_library(self) -> object:
        return self.world.get_blueprint_library()


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


    def distance_to_line(self, A, B, p):
        # This method calculates the perpendicular distance from a point p 
        # to a line defined by two points A and B in a 3D space
        num   = np.linalg.norm(np.cross(B - A, A - p)) # calculate cross product 
        denom = np.linalg.norm(B - A) # Euclidean distance between points A and B
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom


    def vector(self, v):
        # The vector method is a utility function that converts a Carla Location, Vector3D, 
        # or Rotation object to a NumPy array for easier manipulation and calculations.
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])


    def get_discrete_action_space(self):
        # provides a predefined set of discrete steering actions that the agent can take
        action_space = \
            np.array([
            -0.50,
            -0.30,
            -0.10,
            0.0,
            0.10,
            0.30,
            0.50
            ])
        return action_space

    # Main vehicle blueprint method
    # It picks a random color for the vehicle everytime this method is called
    def get_vehicle(self, vehicle_name):
        blueprint = self.blueprint_library.filter(vehicle_name)[0]
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        return blueprint


    # Spawn the vehicle in the environment
    def set_vehicle(self, vehicle_bp, spawn_points):
        # Main vehicle spawned into the env
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)


    # Clean up method
    def remove_sensors(self):
        self.camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None
        self.env_camera_obj = None
        self.front_camera = None
        self.collision_history = None
        self.wrong_maneuver = None

# -------------------------------------------------
# Calculating Euclidean Distance List            |
# -------------------------------------------------

    def calculate_euc_dist(self, current_waypoint):
        # Define the number of closest waypoints to retrieve
        num_closest_waypoints = 10  # You can adjust this as needed

        # Ensure there are at least 10 waypoints available
        if len(self.route_waypoints) < num_closest_waypoints:
            raise ValueError("Not enough waypoints available for calculation.")

        # Retrieve the next 10 waypoints from the beginning of the route_waypoints list
        closest_waypoints = self.route_waypoints[:num_closest_waypoints]

        # Calculate the Euclidean distance to each waypoint
        euclidean_dist_list = []
        for waypoint in closest_waypoints:
            # Extract the coordinates from the waypoint object and convert them into a NumPy array
            waypoint_location = np.array([waypoint.transform.location.x, waypoint.transform.location.y, waypoint.transform.location.z])
            # Calculate the Euclidean distance from the current_waypoint to the waypoint
            distance = np.linalg.norm(np.array(current_waypoint) - waypoint_location)
            # Append the distance to the list
            euclidean_dist_list.append(distance)

        return euclidean_dist_list
    
    def calculate_euc_dist_from_front(self):
        # Define the number of closest waypoints to retrieve
        num_closest_waypoints = 10  # You can adjust this as needed

        # Ensure there are at least 10 waypoints available
        if len(self.route_waypoints) < num_closest_waypoints:
            raise ValueError("Not enough waypoints available for calculation.")

        # Get the position of the front of the vehicle
        vehicle_location = self.vector(self.vehicle.get_location())

        # Retrieve the next 10 waypoints from the beginning of the route_waypoints list
        closest_waypoints = self.route_waypoints[:num_closest_waypoints]

        # Calculate the Euclidean distance from the front of the vehicle to each waypoint
        euclidean_dist_list = []
        for waypoint in closest_waypoints:
            # Extract the coordinates from the waypoint object and convert them into a NumPy array
            waypoint_location = np.array([waypoint.transform.location.x, waypoint.transform.location.y, waypoint.transform.location.z])

            # Calculate the Euclidean distance from the front of the vehicle to the waypoint
            distance = np.linalg.norm(waypoint_location - vehicle_location)

            euclidean_dist_list.append(distance)

        return euclidean_dist_list

# -------------------------------------------------
# Calculating Deviation Angle List               |
# -------------------------------------------------
    def calculate_deviation_angle_cos(self):
        # Define the number of closest waypoints to retrieve
        num_closest_waypoints = 10  # You can adjust this as needed

        # Ensure there are at least 10 waypoints available
        if len(self.route_waypoints) < num_closest_waypoints:
            raise ValueError("Not enough waypoints available for calculation.")

        # Get the forward vector of the vehicle
        vehicle_forward_vector = self.vector(self.vehicle.get_transform().rotation.get_forward_vector())[:2]

        # Get the positions of the nearest 10 waypoints
        nearest_waypoints = self.route_waypoints[:num_closest_waypoints]

        deviation_angles = []

        for waypoint in nearest_waypoints:
            # Calculate the direction vector from the vehicle to the waypoint
            waypoint_vector = self.vector(waypoint.transform.location) - self.vector(self.vehicle.get_location())[:2]

            # Calculate the angle between the vehicle's forward vector and the direction to the waypoint
            deviation_angle = np.arccos(np.dot(vehicle_forward_vector, waypoint_vector) /
                                         (np.linalg.norm(vehicle_forward_vector) * np.linalg.norm(waypoint_vector)))
            
            # Ensure the angle is within the range [-π, π]
            if deviation_angle > np.pi:
                deviation_angle -= 2 * np.pi
            elif deviation_angle <= -np.pi:
                deviation_angle += 2 * np.pi

            deviation_angles.append(deviation_angle)

        return deviation_angles
    
    def calculate_deviation_angle_tan(self):
        # Define the number of closest waypoints to retrieve
        num_closest_waypoints = 10  # You can adjust this as needed

        # Ensure there are at least 10 waypoints available
        if len(self.route_waypoints) < num_closest_waypoints:
            raise ValueError("Not enough waypoints available for calculation.")

        # Get the forward vector of the vehicle
        vehicle_forward_vector = self.vector(self.vehicle.get_transform().rotation.get_forward_vector())[:2]

        # Get the positions of the nearest 10 waypoints
        nearest_waypoints = self.route_waypoints[:num_closest_waypoints]

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
    
    # -------------------------------------------------
    # Estimating Engine RPM                          |
    # -------------------------------------------------
    def estimate_engine_rpm(self):
        # Get the control input for the vehicle
        control = self.vehicle.get_control()
        physics = self.vehicle.get_physics_control()

        # Calculate engine RPM
        engine_rpm = physics.max_rpm * control.throttle

        # Check if the vehicle is in gear (not in neutral or reverse)
        if control.gear > 0:
            # Retrieve the gear information for the current gear
            gear = physics.forward_gears[control.gear]

            # Adjust engine RPM based on the gear ratio of the current gear
            engine_rpm *= gear.ratio

        return engine_rpm
    
    def reward_function(lateral_deviation, deviation_angle, velocity):
        # Calculate the reward based on lateral deviation, heading error, and velocity.
        # Parameters:
        # lateral_deviation (float): The lateral distance from the centerline.
        # heading_error (float): The difference in the vehicle's heading and the road's heading.
        # velocity (float): The current velocity of the vehicle.
        
        # Returns:
        # float: The calculated reward.

        # Note: constants are defined in settings.py
        
        # Normalize the deviation and error
        norm_lateral_deviation = min(abs(lateral_deviation) / MAX_DEVIATION_DISTANCE, 1)
        norm_heading_error = min(abs(deviation_angle) / MAX_DEVIATION_ANGLE, 1)
        norm_velocity_diff = min(abs(DESIRED_VELOCITY - velocity) / MAX_VELOCITY_DIFF, 1)

        # Calculate the reward components
        reward_lateral_deviation = (1 - K_d * norm_lateral_deviation)
        reward_heading = (1 - K_h * norm_heading_error)
        reward_velocity = (1 - K_v * norm_velocity_diff)

        # Combine the rewards
        #total_reward = reward_lateral_deviation * reward_heading * reward_velocity
        numerator = (MAX_DEVIATION_DISTANCE - lateral_deviation) * (MAX_DEVIATION_ANGLE - deviation_angle)
        denominator = MAX_DEVIATION_DISTANCE * MAX_DEVIATION_ANGLE
        total_reward = (numerator/denominator) * (velocity/MAX_VELOCITY_THRESHOLD) * REWARD_CONSTANT_C

        return total_reward
    
        # Example usage of reward_function
        #lateral_deviation_example = 0.5  # example lateral deviation
        #heading_error_example = np.pi / 6  # example heading error
        #velocity_example = 8  # example velocity

        #reward = reward_function(lateral_deviation_example, heading_error_example, velocity_example)
        #print("Calculated Reward:", reward)

    
