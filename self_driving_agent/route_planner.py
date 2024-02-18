import carla

class CarlaRoutePlanner:
    def __init__(self, vehicle, total_distance, town):
        self.vehicle = vehicle
        self.map = self.vehicle.get_world().get_map()
        self.total_distance = total_distance
        self.town = town
        self.route_waypoints = []

    def generate_route(self):
        self.route_waypoints = []  # Initialize an empty list for the route waypoints
        self.current_waypoint_index = 0

        # Get the initial waypoint based on the vehicle's current location
        current_waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        self.route_waypoints.append(current_waypoint)

        for x in range(self.total_distance):
            if x < 650:
                next_waypoint = current_waypoint.next(5.0)[0]
            else:
                next_waypoint = current_waypoint.next(5.0)[-1]

            self.route_waypoints.append(next_waypoint)
            current_waypoint = next_waypoint

    def get_next_waypoint(self):
        if self.current_waypoint_index < len(self.route_waypoints):
            next_waypoint = self.route_waypoints[self.current_waypoint_index]
            self.current_waypoint_index += 1
            return next_waypoint
        else:
            return None
