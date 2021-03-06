import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_privacy as tfp
import numpy as np


rdp_orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
              list(range(5, 64)) + [128, 256, 512])

total_clients = 750
base_noise_multiplier = 0.5
base_clients_per_round = 10
target_delta = 1e-5
target_eps = 2
rounds=200


def get_epsilon(clients_per_round):
    # If we use this number of clients per round and proportionally
    # scale up the noise multiplier, what epsilon do we achieve?
    q = clients_per_round / total_clients
    noise_multiplier = base_noise_multiplier
    noise_multiplier *= clients_per_round / base_clients_per_round
    rdp = tfp.compute_rdp(
        q, noise_multiplier=noise_multiplier, steps=rounds, orders=rdp_orders)
    eps, _, _ = tfp.get_privacy_spent(rdp_orders, rdp, target_delta=target_delta)
    return clients_per_round, eps, noise_multiplier


def find_needed_clients_per_round():
    hi = get_epsilon(base_clients_per_round)
    if hi[1] < target_eps:
        return hi

    # Grow interval exponentially until target_eps is exceeded.
    while True:
        lo = hi
        hi = get_epsilon(2 * lo[0])
        if hi[1] < target_eps:
            break

    # Binary search.
    while hi[0] - lo[0] > 1:
        mid = get_epsilon((lo[0] + hi[0]) // 2)
        if mid[1] > target_eps:
            lo = mid
        else:
            hi = mid

    return hi


clients_per_round, _, noise_multiplier = find_needed_clients_per_round()
print(f'To get ({target_eps}, {target_delta})-DP, use {clients_per_round} '
      f'clients with noise multiplier {noise_multiplier}.')