
import operator
import numpy as np
import matplotlib
matplotlib.use('TKAgg')

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from FL import FLData, FLTraining, FLDataSelector
from mobility import RWPNode, World, generate_rpgm_mobility


class RandEvent:
    """A class representing a randomly generated event, as well as
    static methods to generate a timeline of events following a
    Poisson process."""

    def __init__(self):
        self.center = None
        self.radius = None
        self.time = None
        self.duration = None
        self.data_slice = None

    def current(self, time):
        if self.time is None or self.duration is None:
            return False

        return self.time <= time and time <= self.time + self.duration

    # Returns a data item from the event's data slice
    def observe_data(self):
        if self.data_slice is None:
            return None

        return self.data_slice

    def __str__(self):
        return "center:{}, radius:{}, time:{}, duration:{}, data_slice:{}".format(
            self.center, self.radius, self.time, self.duration, self.data_slice)

    @staticmethod
    def generate_events_timeline(
            timelimit,
            rate,
            ev_size,
            ev_duration,
            world,
            fed_data_selector):
        # Poisson process
        events = []
        time = 0

        max_duration = 0

        while time < timelimit:
            next_time = -np.log(1.0 - np.random.uniform()) / rate
            time += next_time

            # Generate event
            next_event = RandEvent()
            next_event.center = world.random_point()
            next_event.radius = np.random.normal(loc=ev_size, scale=1.0)
            next_event.duration = np.random.normal(loc=ev_duration, scale=10.0)
            next_event.data_slice = fed_data_selector.get_dataslice(
                next_event.center[0], next_event.center[1])
            next_event.time = time

            if (next_event.duration > max_duration):
                max_duration = next_event.duration

            events.append((time, next_event))

        return (events, max_duration)

    # Returns whether "vehicle" matches "event"
    @staticmethod
    def match_event(vehicle, event, time):
        return event.current(time) and World.distance(
            vehicle.mobility.position, event.center) <= event.radius

    # Returns whether "vehicle" matches at least one event from "events"

    @staticmethod
    def match_events(vehicle, events, time):
        for e in events:
            if RandEvent.match_event(vehicle, e, time):
                #print ("match! vid:{}, vpos:{}, vspeed:{} e:{}".format(vehicle.v_id, vehicle.position, vehicle._speed, e))
                return e

        return None

    # Finds the index right before (if smallerIndex == True) or after "value"
    # in "l[start:end]"
    @staticmethod
    def find_ix_sorted(l, value, start, end, smaller_index):

        if start == end - 1:
            return start if smaller_index else end

        mid_point = (start + end) // 2
        if l[mid_point] == value:
            return mid_point
        elif l[mid_point] < value:
            return RandEvent.find_ix_sorted(
                l, value, mid_point, end, smaller_index)
        elif l[mid_point] > value:
            return RandEvent.find_ix_sorted(
                l, value, start, mid_point, smaller_index)

    # Return the list of events happening between start_time and end_time
    # (excluded)
    @staticmethod
    def select_events_between(events, start_time, end_time):

        times = list((time for (time, _) in events))

        begin = RandEvent.find_ix_sorted(
            times, start_time, 0, len(times), True)
        end = RandEvent.find_ix_sorted(times, end_time, 0, len(times), False)

        return events[begin + 1:end]


DB_DISCARD_TIME = 300
DB_DISCARD_THRESHOLD = 500


class Vehicle:
    """ A class representing a vehicle. The vehicle is able to move within the bounds
    described by a World instance, following a (steady-state) Random Waypoint mobility model."""

    def __init__(self, mobility_model, v_id):

        self.v_id = v_id
        self.mobility = mobility_model

        # "Database"
        self.db_last_obs = None

    def __eq__(self, other):
        return (self.v_id == other.v_id)

    def drive(self, timestep):
        self.mobility.move(timestep)

    # Discard old data
    def update_db(self, time):

        if self.db_last_obs is not None:

            # First condition: timeout?
            if self.db_last_obs['time'] + DB_DISCARD_TIME < time:
                self.db_last_obs = None

            # Second condition: out of relevant area?
            elif World.distance(self.mobility.position, self.db_last_obs['position']) > DB_DISCARD_THRESHOLD:
                #print("vid {}: discard interest because of distance:{}".format(self.v_id, World.distance(self.position, self.db_last_obs[1])))
                self.db_last_obs = None

    def rx_model_forstep(self, step_id, network_tx, time):

        #print ("vid {} rx model".format(self.v_id))
        if self.db_last_obs is None:
            print(
                "vid {}: discarding model for step id {} received at time {}, no data".format(
                    self.v_id, step_id, time))
            return

        localtraining_time = np.random.normal(loc=1, scale=0.5)
        network_tx.to_coordinator(
            step_id,
            self.v_id,
            time + localtraining_time,
            self.db_last_obs['federated_data'])
        print(
            "vid {}: model for step id {} received from coordinator at time {}, training and sending back at time {}".format(
                self.v_id,
                step_id,
                time,
                time +
                localtraining_time))

    def observe(self, time, event, network_tx):
        #print ("vid {}, observation at time {} and loc {}".format(self.v_id, time, self.position))

        observed_data = event.observe_data()
        if observed_data is None:
            return

        position_cpy = np.copy(self.mobility.position)
        self.db_last_obs = {
            "time": time,
            "position": position_cpy,
            "federated_data": observed_data}
        network_tx.notify_interest_to_coordinator(
            self.v_id, position_cpy, time, time)

    def observe_with_selector(self, time, fed_selector, network_tx):
        #print ("vid {}, observation at time {} and loc {}".format(self.v_id, time, self.position))

        if self.db_last_obs is None or self.db_last_obs['time'] + 10 <= time:

            position_cpy = np.copy(self.mobility.position)
            self.db_last_obs = {
                "time": time,
                "position": position_cpy,
                "federated_data": fed_selector.get_dataslice(
                    position_cpy[0],
                    position_cpy[1])}
            network_tx.notify_interest_to_coordinator(
                self.v_id, position_cpy, time, time)

    @staticmethod
    def generate_vehicles_RWP(nb_vehicles, world):
        vehicles = []
        for i in range(nb_vehicles):
            vehicles.append(Vehicle(RWPNode(world, i), i))

        return vehicles

    @staticmethod
    def generate_vehicles_RPGM(groups, world):
        vehicles = []
        mobility_nodes = generate_rpgm_mobility(groups, world)

        for node in mobility_nodes:
            vehicles.append(Vehicle(node, node.n_id))

        return vehicles


class NetworkTx:
    """ A simple class to simulate transmission delays of messages between
    the federated learning coordinator and the vehicles.
    Messages to be sent are cached and passed on after a timeout."""

    def __init__(self):

        self.tx_to_vehicle = []
        self.tx_to_coordinator = []
        self.tx_interest_to_coordinator = []

    def notify_interest_to_coordinator(
            self, v_id, location, sense_time, transfer_time):

        tx_time = np.random.normal(loc=1.0, scale=1.0)
        self.tx_interest_to_coordinator.append(
            (transfer_time + tx_time, v_id, location, sense_time))

    # Vehicle transmits model update to coordinator (from step "step_id") at
    # time "transfer_time"
    def to_coordinator(self, step_id, v_id, transfer_time, federated_data):

        tx_time = np.random.normal(loc=1.0, scale=1.0)
        self.tx_to_coordinator.append(
            (transfer_time + tx_time, v_id, step_id, federated_data))

    # Coordinator transmits model for training step "step_id" to "vehicle" at
    # time "transfer_time"
    # Coordinator transmits model (step_id) to "vehicle"
    def to_vehicle(self, vehicle, step_id, transfer_time):

        tx_time = np.random.normal(loc=1.0, scale=1.0)
        self.tx_to_vehicle.append((transfer_time + tx_time, vehicle, step_id))

    def update(self, coordinator, time):
        # Tx coordinator -> vehicle
        to_keep_tovehicle = []
        for (rx_time, vehicle, step_id) in self.tx_to_vehicle:
            if time >= rx_time:
                vehicle.rx_model_forstep(step_id, self, time)
                #print ("time {}, vid {} received model for step {}".format(time, vehicle.v_id, step_id))
            else:
                to_keep_tovehicle.append((rx_time, vehicle, step_id))
        self.tx_to_vehicle = to_keep_tovehicle

        # Tx vehicle -> coordinator
        to_keep_tocoordinator = []
        for (rx_time, v_id, step_id, federated_data) in self.tx_to_coordinator:
            if time >= rx_time:
                coordinator.rx_model_update(
                    v_id, step_id, time, federated_data)
                #print ("time {}, coordinator received update from vid {} for step {}".format(time, v_id, step_id))
            else:
                to_keep_tocoordinator.append(
                    (rx_time, v_id, step_id, federated_data))
        self.tx_to_coordinator = to_keep_tocoordinator

        # Tx vehicle interest -> coordinator
        to_keep_interests = []
        for (
            rx_time,
            v_id,
            location,
                interest_time) in self.tx_interest_to_coordinator:
            if time >= rx_time:
                coordinator.rx_vehicle_interest(
                    v_id, location, interest_time, time)
                #print ("time {}, coordinator received model interest: vid {} time of sensing {}".format(time, v_id, interest_time))
            else:
                to_keep_interests.append(
                    (rx_time, v_id, location, interest_time))
        self.tx_interest_to_coordinator = to_keep_interests


INTEREST_DISCARDTIME = 300


class Coordinator:
    """ A class representing the FL coordinator in charge for training a model
    based on the EMNIST dataset by the FedAvg algorithm. The coordinator initiates
    a new training step every T=15s, in which a set of vehicles is chosen to receive
    and locally train the current model. The coordinator then waits and receives model
    updates back from replying vehicles, before aggregating the results.
    Additionally, the coordinator receives training interests from vehicles,
    that it uses for pertinent client selection in the VKN-assisted approach."""

    def __init__(
            self,
            veh_per_step,
            fed_training,
            fed_data_handler,
            statistics):

        self.statistics = statistics
        self.fed_training = fed_training
        self.fed_data_handler = fed_data_handler

        self.nb_updates_total = 0
        self.modelsent_step = []
        self.updatesreceived_step = []
        self.step_fed_data = []

        self.step_duration = 15
        self.vehicles_per_step = veh_per_step['veh_per_step']
        self.vkn_adapt_vperstep = veh_per_step['vkn_adapt_vperstep']

        self.step_id = -1
        self.step_starttime = -21

        self.vkn_known_interests = {}

    def update(self, vehicles, time, network_tx, vkn):
        if time > self.step_starttime + self.step_duration:

            if self.step_id != -1:

                # Perform federated learning, aggregation of received models at
                # last step.
                print(
                    "step {}: fed data: {}".format(
                        self.step_id,
                        self.step_fed_data))
                step_eval = self.fed_training.training_round(
                    self.fed_data_handler.get_federated_data(self.step_fed_data))

                # Statistics
                if step_eval is not None:
                    A = step_eval['eval']
                    self.statistics.register_training_evaluation(
                        self.step_id, A['sparse_categorical_accuracy'], A['loss'])

                self.statistics.register_step_efficiency(self.step_id, len(
                    self.updatesreceived_step) / self.vehicles_per_step)
                print("Last step: {}/{} models OK".format(
                    len(self.updatesreceived_step), self.vehicles_per_step))

                #print ("Known interests count: {}".format(len(self.vkn_known_interests.keys())))

                # Remove vehicles that did not respond from the interests list
                #print ("self.modelsent_step:{}".format(self.modelsent_step))
                #print ("self.updatesreceived_step:{}".format(self.updatesreceived_step))
                vehicles_noresponse = [
                    v_id for v_id in self.modelsent_step if v_id not in self.updatesreceived_step]
                print("Unresponsive vehicles: {}".format(vehicles_noresponse))

                new_interests = {}
                for (v_id, (interest_time, position)
                     ) in self.vkn_known_interests.items():
                    if v_id not in self.modelsent_step or v_id in self.updatesreceived_step:
                        new_interests[v_id] = (interest_time, position)
                        #print("Keeping interest {}".format(v_id))

                self.vkn_known_interests = new_interests

            self.step_fed_data = []
            self.modelsent_step = []
            self.updatesreceived_step = []

            # Interests update
            interests_to_keep = {}
            for (v_id, (interest_time, position)
                 ) in self.vkn_known_interests.items():
                if interest_time + INTEREST_DISCARDTIME >= time:
                    interests_to_keep[v_id] = (interest_time, position)
            self.vkn_known_interests = interests_to_keep

            if vkn and len(self.vkn_known_interests.keys()) > 0:
                print("Coordinator: new step (vkn) (time={})".format(time))
                self.new_step_vkn(vehicles, time, network_tx)
            else:
                print("Coordinator: new step (time={})".format(time))
                self.new_step_random(vehicles, time, network_tx)

    def new_step_vkn(self, vehicles, time, network_tx):

        self.step_id += 1
        self.step_starttime = time

        selected_indexes = []

        # From interests
        #print ( "interests:{}".format( np.array(list(self.vkn_known_interests.keys())) ) )

        sorted_interests = sorted(
            self.vkn_known_interests.items(),
            key=operator.itemgetter(1),
            reverse=True)
        sorted_vids = list((v_id for (v_id, _) in sorted_interests))

        selected_indexes_interests = np.array(
            sorted_vids[0:np.minimum(len(sorted_vids), self.vehicles_per_step)])
        #selected_indexes_interests = np.array(list(self.vkn_known_interests.keys()))[np.random.choice(len(self.vkn_known_interests), size=np.minimum(len(self.vkn_known_interests), self.vehicles_per_step), replace=False)]
        #print ( "chosen interests: vids: {}".format(selected_indexes_interests) )

        for v_id in selected_indexes_interests:
            selected_indexes.append(v_id)

        nb_random_vehicles = self.vehicles_per_step - len(selected_indexes)

        selected_indexes_tmp = []
        selected_interest_pos = []

        # LESS KNOWN INTERESTS THAN CLIENTS TO SELECT: SELECT ALL INTERESTS +
        # FILL WITH RANDOM
        if len(self.vkn_known_interests.items()) <= self.vehicles_per_step:
            for vid in self.vkn_known_interests.keys():
                selected_indexes.append(vid)

            nb_random_vehicles = self.vehicles_per_step - len(selected_indexes)
            if nb_random_vehicles > 0:
                remaining_indexes = np.array([ix for ix in range(
                    len(vehicles)) if ix not in selected_indexes_interests])
                extra_selected_indexes = remaining_indexes[np.random.choice(
                    len(remaining_indexes), size=nb_random_vehicles, replace=False)]

                for v_id in extra_selected_indexes:
                    selected_indexes.append(v_id)

        else:  # Enough interests: CLUSTERING OF INTERESTS

            interests_vid = []
            interests_times = []
            interests_locations = []
            for (vid, (interest_time, loc)
                 ) in self.vkn_known_interests.items():
                interests_vid.append(vid)
                interests_times.append(interest_time)
                interests_locations.append(loc)

            interests_vid = np.array(interests_vid)
            interests_times = np.array(interests_times)
            interests_locations = np.array(interests_locations)

            kmeans = KMeans(
                n_clusters=self.vehicles_per_step,
                tol=10,
                max_iter=30)
            kmeans.fit(interests_locations)

            # Taking the most recent vehicles by cluster.
            for cluster_id in range(self.vehicles_per_step):
                print("cluster {}".format(cluster_id))

                cluster_point_ids = []
                for j in range(len(interests_vid)):
                    if kmeans.labels_[j] == cluster_id:
                        cluster_point_ids.append(j)

                cluster_point_ids = np.array(cluster_point_ids)
                newest_interest = -1
                newest_time = -1
                newest_location = -1
                for point_id in cluster_point_ids:

                    if newest_interest == - \
                            1 or interests_times[point_id] > newest_time:
                        newest_interest = interests_vid[point_id]
                        newest_time = interests_times[point_id]
                        newest_location = interests_locations[point_id]

                '''locs = interests_locations[cluster_point_ids]
                print (locs)

                plt.scatter(locs[:, 0], locs[:, 1], color='orange')
                plt.scatter(kmeans.cluster_centers_[cluster_id,0], kmeans.cluster_centers_[cluster_id,1], color='green')
                plt.xlim((0,1000))
                plt.ylim((0,1000))
                plt.show()'''

                selected_interest_pos.append(newest_location)
                selected_indexes_tmp.append(newest_interest)

            # Final step: Reduce amount of clients selected if enabled
            remove_indexes = []
            if self.vkn_adapt_vperstep:
                for (i, pos1) in enumerate(selected_interest_pos):

                    finished = False
                    j = i + 1
                    while j < len(selected_interest_pos) and not finished:
                        pos2 = selected_interest_pos[j]
                        if World.distance(pos1, pos2) <= 50:
                            remove_indexes.append(i)
                            finished = True
                        j += 1

            selected_indexes = []
            for (ix, value) in enumerate(selected_indexes_tmp):
                if ix not in remove_indexes:
                    selected_indexes.append(value)

            print("NB CLIENTS SELECTED: {}".format(len(selected_indexes)))

            '''
            debug = np.array(vehicles)[np.array(selected_indexes)]
            selected_vehicles_pos = []
            for v in debug:
                selected_vehicles_pos.append(v.mobility.position)
            selected_vehicles_pos = np.array(selected_vehicles_pos)

            all_vehicles_pos = []
            for v in vehicles:
                all_vehicles_pos.append(v.mobility.position)
            all_vehicles_pos = np.array(all_vehicles_pos)

            random_vehs = np.random.choice(vehicles, 10)
            random_vehicles_pos = []
            for v in random_vehs:
                random_vehicles_pos.append(v.mobility.position)
            random_vehicles_pos = np.array(random_vehicles_pos)

            plt.title("Clustering of training samples' location of sensing by the coordinator")
            plt.scatter(all_vehicles_pos[:, 0], all_vehicles_pos[:, 1], color='gray', alpha=0.3) # All vehicles
            plt.scatter(random_vehicles_pos[:,0], random_vehicles_pos[:,1], color='orange', marker='x') # Random selected v
            plt.scatter(selected_vehicles_pos[:,0], selected_vehicles_pos[:,1], color='blue', marker='x') # VKN selected v
            plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color='purple', marker='$c$') # Clusters
            plt.xlim((0,1000))
            plt.ylim((0,1000))
            plt.show()'''

        #print ("final selected indexes: selected vids:{}".format(selected_indexes))

        self.statistics.register_step_nbselect(
            self.step_id, len(selected_indexes))
        selected_vehicles = np.array(vehicles)[np.array(selected_indexes)]

        print("step_id:{}, sent to vehicles:".format(self.step_id))

        for v in selected_vehicles:
            network_tx.to_vehicle(v, self.step_id, time)
            print(v.v_id)

        self.modelsent_step = selected_indexes

    def new_step_random(self, vehicles, time, network_tx):

        self.step_id += 1
        self.step_starttime = time

        selected_indexes = np.random.choice(
            len(vehicles), size=self.vehicles_per_step, replace=False)
        selected_vehicles = np.array(vehicles)[selected_indexes]
        print("step_id:{}, sent to vehicles:".format(self.step_id))

        for v in selected_vehicles:
            network_tx.to_vehicle(v, self.step_id, time)
            print(v.v_id)

        self.modelsent_step = selected_indexes

    # Reception of a model update from vehicle
    def rx_model_update(self, v_id, step_id, time, federated_data):
        if step_id != self.step_id:  # An old model, discard
            print(
                "time {} vid {} stepid {} != current step id {}: Model discarded".format(
                    time, v_id, step_id, self.step_id))
            return

        print(
            "Model update received at time {}, vid:{} / stepid:{} ; federated data id: {}".format(
                time,
                v_id,
                step_id,
                federated_data))
        self.step_fed_data.append(federated_data)
        self.nb_updates_total += 1
        self.updatesreceived_step.append(v_id)

        self.statistics.register_new_step(time, v_id)

    def rx_vehicle_interest(self, v_id, location, interest_time, rx_time):

        #print ("time {}: Coordinator received interest (vid:{}, location: {}, interest time:{})".format(rx_time, v_id, location, interest_time))
        self.vkn_known_interests[v_id] = (
            interest_time, [location[0], location[1]])


class Statistics:
    """ A simple class to keep track of the progress of a simulation."""

    def __init__(self):
        self.steps_per_time = {}
        self.step_efficiency = {}
        self.step_nbselect = {}

        self.training_accuracy = {}
        self.training_loss = {}

        self.participating_vids = []
        self.newvids_per_time = {}

    def register_new_step(self, time, v_id):

        if time not in self.newvids_per_time:
            self.newvids_per_time[time] = 0

        if v_id not in self.participating_vids:
            self.participating_vids.append(v_id)
            self.newvids_per_time[time] += 1

        if time not in self.steps_per_time:
            self.steps_per_time[time] = 0

        self.steps_per_time[time] += 1

    def register_step_efficiency(self, step_id, rate):
        self.step_efficiency[step_id] = rate

    def register_step_nbselect(self, step_id, nbselect):
        self.step_nbselect[step_id] = nbselect

    def register_training_evaluation(self, step_id, accuracy, loss):
        self.training_accuracy[step_id] = accuracy
        self.training_loss[step_id] = loss

    def get_total_steps_per_time(self):

        # Total steps per time. [0] (time) and [1] (aggregated steps)
        times = []
        total_steps = []

        current_total = 0
        for time in sorted(self.steps_per_time.keys()):
            current_total += self.steps_per_time[time]

            total_steps.append(current_total)
            times.append(time)

        return (times, total_steps)

    def get_cumulated_newvids_per_time(self):

        # Total steps per time. [0] (time) and [1] (aggregated steps)
        times = []
        total_vids = []

        current_total = 0
        for time in sorted(self.newvids_per_time.keys()):
            current_total += self.newvids_per_time[time]

            total_vids.append(current_total)
            times.append(time)

        return (times, total_vids)

    def plot_total_steps_per_time(self):

        (times, total_steps) = self.get_total_steps_per_time()

        plt.plot(times, total_steps)
        plt.show()

    # Smoothed dict {time:value}

    def get_smoothed_list(self, data, smooth):
        x = []
        y = []

        times = list(sorted(data.keys()))

        smooth_value = 0
        smooth_index = 0
        for timeval in times:

            smooth_value += data[timeval]
            smooth_index += 1

            if smooth_index == smooth:
                x.append(timeval)
                y.append(smooth_value / smooth)
                smooth_index = 0
                smooth_value = 0

        return (x, y)

    def get_step_efficiency(self, smooth):
        return self.get_smoothed_list(self.step_efficiency, smooth)

    def get_step_nbselect(self, smooth):
        return self.get_smoothed_list(self.step_nbselect, smooth)

    def get_training_accuracy(self, smooth):
        return self.get_smoothed_list(self.training_accuracy, smooth)

    def get_training_loss(self, smooth):
        return self.get_smoothed_list(self.training_loss, smooth)

    def plot_step_efficiency(self, smooth):
        (x, y) = self.get_step_efficiency(smooth)
        plt.plot(x, y)
        plt.show()

    def plot_training_accuracy(self, smooth):
        (x, y) = self.get_training_accuracy(smooth)
        plt.title("Training accuracy")
        plt.plot(x, y)
        plt.show()

    def plot_training_loss(self, smooth):
        (x, y) = self.get_training_loss(smooth)
        plt.title("Training loss")
        plt.plot(x, y)
        plt.show()


class Simulation:
    """ A class to initialize and run a simulation."""

    def __init__(
            self,
            size,
            nb_vehicles,
            mobility,
            event_conf,
            timelimit,
            timestep,
            veh_per_step,
            vkn,
            seed):

        np.random.seed(seed)
        FLTraining.set_seed(seed)

        self.time = 0
        self.timelimit = timelimit
        self.timestep = timestep
        self.vkn = vkn
        self.veh_per_step = veh_per_step

        self.world = World(size)

        self.vehicles = []
        if mobility['model'] == "RPGM" and 'groups' in mobility:
            self.vehicles = Vehicle.generate_vehicles_RPGM(
                mobility['groups'], self.world)
            print("Using RPGM model: v:{} groups:{}".format(
                sum(mobility['groups']), mobility['groups']))

        else:  # Random Waypoint
            self.vehicles = Vehicle.generate_vehicles_RWP(
                nb_vehicles, self.world)

        self.fed_data = FLData()
        self.fed_training = FLTraining(
            self.fed_data.get_federated_test_data(
                self.fed_data.get_all_test_samples_id()[
                    0:20]))

        # Events generation
        use_events = event_conf['use_events']
        event_data_distribution = event_conf['data_distribution']
        cut = 0
        if 'cut' in event_conf:
            cut = event_conf['cut']

        self.fed_data_selector = FLDataSelector(
            self.world.dimension, event_data_distribution, self.fed_data, cut)
        (self.events, self.events_max_duration) = (None, None)

        if use_events:
            event_rate = event_conf['rate']
            event_size = event_conf['size']
            event_duration = event_conf['duration']
            (self.events, self.events_max_duration) = RandEvent.generate_events_timeline(
                self.timelimit, event_rate, event_size, event_duration, self.world, self.fed_data_selector)

        self.statistics = Statistics()
        self.coordinator = Coordinator(
            self.veh_per_step,
            self.fed_training,
            self.fed_data,
            self.statistics)
        self.network_tx = NetworkTx()

    def step(self):

        self.time += self.timestep

        # Currently happening events
        current_events = None
        if self.events is not None:
            current_events = RandEvent.select_events_between(
                self.events, self.time - self.events_max_duration - 0.01, self.time + 0.01)

        # Drive vehicles
        for v in self.vehicles:
            v.drive(self.timestep)
            v.update_db(self.time)

            # Match events
            if current_events is None:
                v.observe_with_selector(
                    self.time, self.fed_data_selector, self.network_tx)
            else:
                ev = RandEvent.match_events(
                    v, (ev for (_, ev) in current_events), self.time)
                if ev is not None:
                    v.observe(self.time, ev, self.network_tx)

        #print ("time: {}".format(self.time))
        self.coordinator.update(
            self.vehicles,
            self.time,
            self.network_tx,
            vkn=self.vkn)
        self.network_tx.update(self.coordinator, self.time)

        return self.time

    def done(self):
        return self.time >= self.timelimit