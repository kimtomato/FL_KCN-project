
"""
 2020 Duncan Deveaux <deveaux@eurecom.fr>
"""

import os
import sys
import argparse
import pickle

from Simulation import Simulation

class NoStdStreams(object):
    def __init__(self, stdout=None, stderr=None):
        self.devnull = open(os.devnull, 'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()


parser = argparse.ArgumentParser(description='Run FL/VKN simulations.')
parser.add_argument(
    '--mobility',
    default="RWP",
    help='="RWP" or "RPGM". Generate simulation results using Random Waypoint or RPGM mobility.')
parser.add_argument(
    '--seed_begin',
    type=int,
    default=0,
    help='Simulations will be generated for random seeds [seed_begin;seed_end[.')
parser.add_argument(
    '--seed_end',
    type=int,
    default=1,
    help='Simulations will be generated for random seeds [seed_begin;seed_end[.')

args = parser.parse_args()

# Simulation's input parameters
events_conf = {}
mob = {}
training = {}
mobility_name = ""
vkn_adapt_vperstep = False

vps = 10  # Number of training vehicles per step

if args.mobility == "RWP":

    events_conf = {
        "use_events": True,
        "rate": 1.5 / 60.0,
        "size": 50.0,
        "duration": 60.0,
        "data_distribution": "random"}
    mob = {"model": 'RWP'}
    training = {"veh_per_step": 10, "vkn_adapt_vperstep": False}

    mobility_name = "rwp"
    vkn_adapt_vperstep = False

else:
    vkn_adapt_vperstep = True
    events_conf = {
        "use_events": False,
        "data_distribution": "localized",
        "cut": 10}
    mob = {"model": 'RPGM', "groups": [650, 20, 20, 10, 10, 10, 10, 10, 5, 5]}
    training = {"veh_per_step": vps, "vkn_adapt_vperstep": True}

    mobility_name = "rpgm"
    vkn_adapt_vperstep = True


stats_vkn = {}
stats_tradi = {}

stats_vkn[vps] = []
stats_tradi[vps] = []


for i in range(args.seed_begin, args.seed_end):

    print("New simulations with seed: {}".format(i))

    simvkn = Simulation(
        size=1000,
        nb_vehicles=750,
        mobility=mob,
        event_conf=events_conf,
        timelimit=3600,
        timestep=1,
        veh_per_step=training,
        vkn=True,
        seed=i)
    simtradi = Simulation(
        size=1000,
        nb_vehicles=750,
        mobility=mob,
        event_conf=events_conf,
        timelimit=3600,
        timestep=1,
        veh_per_step=training,
        vkn=False,
        seed=i)

    print("\tVKN simulation...")
    while not simvkn.done():
        with NoStdStreams():
            time = simvkn.step()
        print("time: {}/3600s".format(time), end="\r")
    stats_vkn[vps].append(simvkn.statistics)

    print("\tvkn ok")
    print("\tTraditional simulation...")
    while not simtradi.done():
        with NoStdStreams():
            time = simtradi.step()
        print("time: {}/3600s".format(time), end="\r")
    stats_tradi[vps].append(simtradi.statistics)


prefix = ""
if vkn_adapt_vperstep:
    prefix = "optimized_"

stats = {'vkn': stats_vkn, 'tradi': stats_tradi}
filename = "{}{}_dump{}_{}_{}".format(
    prefix,
    mobility_name,
    vps,
    args.seed_begin,
    args.seed_end)
with open(filename, 'wb') as filehandler:
    pickle.dump(stats, filehandler)

print('done.')
