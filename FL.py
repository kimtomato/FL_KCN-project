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



def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax(),
    ])


emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0])
preprocessed_example_dataset = preprocess(example_dataset)


def model_fn():
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=preprocessed_example_dataset.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


evaluation = tff.learning.build_federated_evaluation(model_fn)


class FLData:

    def __init__(self):
        self.emnist_train, self.emnist_test = tff.simulation.datasets.emnist.load_data()

    def get_nb_samples(self):
        return len(self.emnist_train.client_ids)

    def get_data_samples_id(self, nb_samples):
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
        if mode == "localized":
            print('SELECTOR: {}x{} cells'.format(cut, cut))

        self.dim = float(worlddim)
        self.fl_data = fl_data
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

    def __init__(self, validation_set):
        # federated averaging
        self.iterative_process = tff.learning.build_federated_averaging_process(
            model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),    # Client - only local update computation
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD( learning_rate=1.0))

        self.state = self.iterative_process.initialize()
        self.validation_set = validation_set
        self.round_id = 0

        self.current_evaluation = None

    def training_round(self, federated_data):

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
