import numpy as np

PI = 3.14159


class World:
    """A simple class holding information about the size of the area containing all the
    vehicles and helper methods to compute points and distances."""

    def __init__(self, dimension):
        self.dimension = dimension

    def random_point(self):
        return np.random.uniform(high=self.dimension, size=2)

    @staticmethod
    def random_point_within_circle(center, radius):
        r = np.random.uniform() * radius
        angle = np.random.uniform() * 2 * PI

        return center + r * np.array((np.cos(angle), np.sin(angle)))

    @staticmethod
    def distance(a, b):
        return np.linalg.norm(a - b)

    @staticmethod
    def intermediate_point(a, b, distance):
        (dy, dx) = (b[1] - a[1], b[0] - a[0])
        angle = np.math.atan2(dy, dx)

        return a + distance * np.array((np.cos(angle), np.sin(angle)))


class MovingNode:
    """ A class representing an abstract moving node. The vehicle is able to move within the bounds
    described by a World instance, following a (steady-state) Random Waypoint mobility model."""

    def __init__(self, world, node_id):
        self.n_id = node_id
        self.position = None

    def __eq__(self, other):
        return (self.n_id == other.n_id)

    def move(self, timestep):
        pass


class RWPNode(MovingNode):
    def __init__(self, world, node_id):
        self.n_id = node_id
        self.position = None
        self.world = world

        self.pause_probability = 0.5
        self.pause_max = 300

        self.min_speed = 5
        self.max_speed = 20

        self._init_initial_path()

    def __eq__(self, other):
        return (self.n_id == other.n_id)

    def _init_initial_path(self):
        # Will the node begin in a paused state ?
        u_pause = np.random.uniform()
        if u_pause < self.pause_probability:

            u = np.random.uniform()
            self.current_pause_time = self.pause_max * (1 - np.sqrt(1 - u))
            self.position = np.random.uniform(size=2) * self.world.dimension
            self._destination = self.position  # Will be recomputed at the end of the pause
            self._speed = 0

            # print ('node id {}: starting in a pause of {}s at pos {}'.format(self.n_id, self.current_pause_time, self.position))

        else:  # The node begins in movement

            self.current_pause_time = 0

            # Initial speed (steady-state)
            u = np.random.uniform()
            self._speed = (self.max_speed ** u) / (self.min_speed ** (u - 1))

            # Initial path (steady-state)
            path_chosen = False
            while not path_chosen:
                (x1, y1) = np.random.uniform(size=2)
                (x2, y2) = np.random.uniform(size=2)

                r = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / np.sqrt(2)
                u1 = np.random.uniform()

                if u1 < r:
                    u2 = np.random.uniform()

                    self.position = np.array(
                        (u2 * x1 + (1 - u2) * x2, u2 * y1 + (1 - u2) * y2)) * self.world.dimension
                    self._destination = np.array(
                        (x2, y2)) * self.world.dimension
                    path_chosen = True

            # print ("node id {}: starting in move: pos:{}, dest:{}, speed:{}".format(self.n_id, self.position, self._destination, self._speed))

    def _compute_new_pause(self):

        # print ("node id {}: computing new potential pause ?".format(self.n_id))
        u_pause = np.random.uniform()

        if u_pause < self.pause_probability:
            u_pausetime = np.random.uniform()
            self.current_pause_time = u_pausetime * self.pause_max
            # print ("nid {}: there will be a pause of {}s".format(self.n_id, self.current_pause_time))

        else:  # No pause: compute new move
            self.current_pause_time = 0
            # print ("nid {}: there will be no pause".format(self.n_id))
            self._compute_new_path()

    def _compute_new_path(self):

        # Path
        self.position = self._destination
        self._destination = np.random.uniform(size=2) * self.world.dimension

        # Speed
        u = np.random.uniform()
        self._speed = self.min_speed + u * (self.max_speed - self.min_speed)

        # print ("nid {}: computed new path: current pos:{}, dest:{}, speed:{}".format(self.n_id, self.position, self._destination, self._speed))

    def move(self, timestep):

        # State: pause ?
        if self.current_pause_time > 0:
            self.current_pause_time = np.maximum(
                0, self.current_pause_time - timestep)
            # print ("nid {}: pausing at pos {}... remaining pause after this step:{}".format(self.n_id, self.position, self.current_pause_time))

            if self.current_pause_time == 0:  # Pause finished
                self._compute_new_path()
            return

        # State: driving
        distance_to_dest = World.distance(self.position, self._destination)
        driven_distance = self._speed * float(timestep)

        # print ("driven distance:{}, dist to dest:{}".format(driven_distance, distance_to_dest))

        if driven_distance >= distance_to_dest:
            # print("recomputing")
            self.position = self._destination
            # print ("nid {} arrived at destination {}={}".format(self.n_id, self.position, self._destination))

            self._compute_new_pause()

        else:  # Continue the drive to waypoint
            self.position = World.intermediate_point(
                self.position, self._destination, driven_distance)
            # print ("nid {}: driving: pos:{}, dest:{}, new dist to dest:{}".format(self.n_id, self.position, self._destination, World.distance(self.position, self._destination)))

    # Helper function: useful for eventual RPGM Following nodes.

    def get_reference_point_area(self):
        return {'pos': self.position, 'radius': 25}

    def get_max_speed(self):
        return self.max_speed

    def is_pausing(self):
        return (self.current_pause_time > 0)

    def get_angle(self):
        a = self.position
        b = self._destination

        (dy, dx) = (b[1] - a[1], b[0] - a[0])
        return np.math.atan2(dy, dx)


class RPGMFollowingNode(MovingNode):
    ''' Node that follows a group leader (the group leader follows steady-state RWP) '''

    def __init__(self, leader, node_id):
        self.n_id = node_id
        self.leader = None

        self.reference_point = None
        self.ref_dist = None
        self.set_group_leader(leader)

    def __eq__(self, other):
        return (self.n_id == other.n_id)

    def set_group_leader(self, leader):
        self.leader = leader
        ref = self.leader.get_reference_point_area()

        # Set initial position
        self.reference_point = World.random_point_within_circle(ref['pos'], 25)
        self.ref_dist = (
            self.reference_point[0] -
            self.leader.position[0],
            self.reference_point[1] -
            self.leader.position[1])

        self.position = World.random_point_within_circle(
            self.reference_point, 25)

    def move(self, timestep):
        # Follow the leader
        if self.leader.is_pausing():
            return

        # Compute new position based on speed and angle
        self.reference_point = np.array(
            [self.leader.position[0] + self.ref_dist[0], self.leader.position[1] + self.ref_dist[1]])
        self.position = World.random_point_within_circle(
            self.reference_point, 25)

        # self.update_speed_angle()
        # self.position += timestep*self.speed*np.array((np.cos(self.angle), np.sin(self.angle)))

        # ref = self.leader.get_reference_point()
        # self.position = World.random_point_within_circle(self.leader.position, 50)

        # print ('v{}, speed:{}, angle:{}'.format(self.n_id, self.speed, self.angle))


# vehicles_count must be divisible by groups_count
def generate_rpgm_mobility(groups_list, world):
    nodes = []

    i = 0
    for group_size in groups_list:

        leader = RWPNode(world, i)
        nodes.append(leader)
        # print("leader {}".format(i))

        for _ in range(group_size - 1):
            i += 1
            follower = RPGMFollowingNode(leader, i)
            nodes.append(follower)
            # print("follower {}".format(i))

        i += 1

    return nodes

