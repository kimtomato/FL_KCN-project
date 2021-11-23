import collections

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_privacy as tfp
import matplotlib.pyplot as plt
import seaborn as sns0



#data download & preprocessing
def get_emnist_dataset():
  emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
      only_digits=True)


  def element_fn(element):
    return collections.OrderedDict(
        x=tf.expand_dims(element['pixels'], -1), y=element['label'])

  def preprocess_train_dataset(dataset):
    # Use buffer_size same as the maximum client dataset size,
    # 418 for Federated EMNIST
    return (dataset.map(element_fn)
                   .shuffle(buffer_size=418)
                   .repeat(1)
                   .batch(32, drop_remainder=False))

  def preprocess_test_dataset(dataset):
    return dataset.map(element_fn).batch(128, drop_remainder=False)

  emnist_train = emnist_train.preprocess(preprocess_train_dataset)
  emnist_test = preprocess_test_dataset(
      emnist_test.create_tf_dataset_from_all_clients())
  return emnist_train, emnist_test

train_data, test_data = get_emnist_dataset()

#model define
def my_model_fn():
  model = tf.keras.models.Sequential([
      tf.keras.layers.Reshape(input_shape=(28, 28, 1), target_shape=(28 * 28,)),
      tf.keras.layers.Dense(200, activation=tf.nn.relu),
      tf.keras.layers.Dense(200, activation=tf.nn.relu),
      tf.keras.layers.Dense(10)])
  return tff.learning.from_keras_model(
      keras_model=model,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      input_spec=test_data.element_spec,
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

#noise determine
# Run five clients per thread. Increase this if your runtime is running out of
# memory. Decrease it if you have the resources and want to speed up execution.
tff.backends.native.set_local_python_execution_context(clients_per_thread=5)

total_clients = len(train_data.client_ids)

def train(rounds, noise_multiplier, clients_per_round, data_frame):
  # Using the `dp_aggregator` here turns on differential privacy with adaptive
  # clipping.
  aggregation_factory = tff.learning.model_update_aggregator.dp_aggregator(
      noise_multiplier, clients_per_round)

  # We use Poisson subsampling which gives slightly tighter privacy guarantees
  # compared to having a fixed number of clients per round. The actual number of
  # clients per round is stochastic with mean clients_per_round.
  sampling_prob = clients_per_round / total_clients

  # Build a federated averaging process.
  # Typically a non-adaptive server optimizer is used because the noise in the
  # updates can cause the second moment accumulators to become very large
  # prematurely.
  learning_process = tff.learning.build_federated_averaging_process(
        my_model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.01),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0, momentum=0.9),
        model_update_aggregation_factory=aggregation_factory)

  eval_process = tff.learning.build_federated_evaluation(my_model_fn)

  # Training loop.
  state = learning_process.initialize()
  for round in range(rounds):
    if round % 5== 0:
      metrics = eval_process(state.model, [test_data])['eval']
      if round < 25 or round % 25 == 0:
        print(f'Round {round:3d}: {metrics}')
      data_frame = data_frame.append({'Round': round,
                                      'NoiseMultiplier': noise_multiplier,
                                      **metrics}, ignore_index=True)

    # Sample clients for a round. Note that if your dataset is large and
    # sampling_prob is small, it would be faster to use gap sampling.
    x = np.random.uniform(size=total_clients)
    sampled_clients = [
        train_data.client_ids[i] for i in range(total_clients)
        if x[i] < sampling_prob]
    sampled_train_data = [
        train_data.create_tf_dataset_for_client(client)
        for client in sampled_clients]

    # Use selected clients for update.
    state, metrics = learning_process.next(state, sampled_train_data)

  metrics = eval_process(state.model, [test_data])['eval']
  print(f'Round {rounds:3d}: {metrics}')
  data_frame = data_frame.append({'Round': rounds,
                                  'NoiseMultiplier': noise_multiplier,
                                  **metrics}, ignore_index=True)

  return data_frame

data_frame = pd.DataFrame()
rounds = 100
clients_per_round = 120
noise_multiplier = 1.2
data_frame = train(rounds, noise_multiplier, clients_per_round, data_frame)

# for noise_multiplier in [0.0]:
#   print(f'Starting training with noise multiplier: {noise_multiplier}')
#   data_frame = train(rounds, noise_multiplier, clients_per_round, data_frame)
#   print()

#loss visualization
def make_plot(data_frame):
  plt.figure(figsize=(15, 5))

  dff = data_frame.rename(
      columns={'sparse_categorical_accuracy': 'Accuracy', 'loss': 'Loss'})
  print(dff)

  plt.subplot(121)
  sns.lineplot(data=dff, x='Round', y='Accuracy', hue='NoiseMultiplier', palette='dark')
  plt.subplot(122)
  sns.lineplot(data=dff, x='Round', y='Loss', hue='NoiseMultiplier', palette='dark')
  plt.show()

make_plot(data_frame)






#######

import collections
import random

import sys

import tensorflow as tf
import tensorflow_federated as tff

NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10


# image preprocess
def preprocess(dataset):

    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        return collections.OrderedDict(
            x=tf.reshape(element['pixels'], [-1, 784]),
            y=tf.reshape(element['label'], [-1, 1]))

    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
        BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


def make_federated_data(client_data, client_ids):
    return [
        preprocess(client_data.create_tf_dataset_for_client(x))
        for x in client_ids
    ]



emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0])
preprocessed_example_dataset = preprocess(example_dataset)

#model define
def model_fn():
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax(),
    ])

    return tff.learning.from_keras_model(
        keras_model=model,
        input_spec=preprocessed_example_dataset.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


evaluation = tff.learning.build_federated_evaluation(model_fn)



#produce data
class FLData:

    def __init__(self):
        self.emnist_train, self.emnist_test = tff.simulation.datasets.emnist.load_data()

    def get_nb_samples(self):
        return len(self.emnist_train.client_ids)

    def get_data_samples_id(self, nb_samples): # the number of nb_sample
        return random.choices(self.emnist_train.client_ids, k=nb_samples)

    def get_federated_data(self, sample_ids):
        return make_federated_data(self.emnist_train, sample_ids)

    def get_federated_test_data(self, sample_ids):
        return make_federated_data(self.emnist_test, sample_ids)

    def get_all_samples_id(self):
        return self.emnist_train.client_ids

    def get_all_test_samples_id(self):
        return self.emnist_test.client_ids


class FLDataSelector:

    def __init__(self, worlddim, mode, fl_data, cut):

        self.div = cut
        #RPG
        if mode == "localized":
            print('SELECTOR: {}x{} cells'.format(cut, cut))

        self.dim = float(worlddim)
        self.fl_data = fl_data
        #PWP
        if mode == "random" or mode == "localized":
            self.mode = mode
        else:
            self.mode = "random"

        self.slices = None
        if mode == "localized":
            # Generate location-based data.
            size = self.div
            self.slices = []
            for i in range(size):
                self.slices.append(self.fl_data.get_data_samples_id(size))

    def get_dataslice(self, x, y):

        if self.mode == "localized":

            x_ = min(max(x, 0), self.dim - 0.01)
            y_ = min(max(y, 0), self.dim - 0.01)

            #print (x_,y_)

            i = int(x_ / self.dim * self.div)
            j = int(y_ / self.dim * self.div)

            #print (i,j)
            #print (self.slices[i][j])
            return self.slices[i][j]

        else:
            return self.fl_data.get_data_samples_id(1)[0]


class FLTraining:
    @staticmethod
    def set_seed(seed):
        tf.random.set_seed(seed)

    tff.backends.native.set_local_python_execution_context(clients_per_thread=5)


    def __init__(self, validation_set,veh_per_step):
        a = veh_per_step['veh_per_step']
        # federated averaging


        aggregation_factory = tff.learning.model_update_aggregator.dp_aggregator(
            0.25, a)



        self.iterative_process = tff.learning.build_federated_averaging_process(
            model_fn,
            #client_optimaizer => only local update computation
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),

            #server_oprimizer => average update adapplication
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD( learning_rate=1.0),
            model_update_aggregation_factory=aggregation_factory
        )

        #server state construct
        self.state = self.iterative_process.initialize()
        self.validation_set = validation_set
        self.round_id = 0

        self.current_evaluation = None

    def training_round(self, federated_data):

        #data X -> SKIP
        if len(federated_data) == 0:
            print('round {:2d}, skip.'.format(self.round_id))

        else:
            self.state, _ = self.iterative_process.next(
                self.state, federated_data)
            self.current_evaluation = evaluation(
                self.state.model, self.validation_set)
            print('round {:2d}, eval_metrics={}'.format(
                self.round_id, self.current_evaluation))

        print("size:{}".format(sys.getsizeof(self.state.model)))

        self.round_id += 1
        return self.current_evaluation




