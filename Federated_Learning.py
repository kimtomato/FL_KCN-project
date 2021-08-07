"""
2020 Duncan Deveaux <deveaux@eurecom.fr> 논문 구현 코드
"""
# 수정본
import collections
import random

import sys

import tensorflow as tf
import tensorflow_federated as tff

NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10


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

        self.iterative_process = tff.learning.build_federated_averaging_process(
            model_fn, client_optimizer_fn=lambda: tf.keras.optimizers.SGD(
                learning_rate=0.02), server_optimizer_fn=lambda: tf.keras.optimizers.SGD(
                learning_rate=1.0))

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


'''
iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))


logdir = "logs/training/"
summary_writer = tf.summary.create_file_writer(logdir)
state = iterative_process.initialize()


sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

federated_train_data = make_federated_data(emnist_train, sample_clients)
print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
if len(federated_train_data) > 0:
    print('First dataset: {d}'.format(d=federated_train_data[0]))


evaluation = tff.learning.build_federated_evaluation(model_fn)

x = []
y1 = []
Y1 = []
y2 = []
Y2 = []


NUM_ROUNDS = 80
#with summary_writer.as_default():
for round_num in range(1, NUM_ROUNDS):

    clients_set = random.choices(emnist_train.client_ids, k=10)
    print("round {} clients:{}".format(round_num, clients_set))
    federated_train_data = make_federated_data(emnist_train, clients_set)
    federated_test_data = make_federated_data(emnist_test, emnist_train.client_ids[0:50])

    state, metrics = iterative_process.next(state, federated_train_data)

    eval_metrics = evaluation(state.model, federated_test_data)
    print('round {:2d}, eval_metrics={}'.format(round_num, eval_metrics))

    x.append(round_num)
    y1.append(eval_metrics['loss'])
    Y1.append(eval_metrics['sparse_categorical_accuracy'])

    #for name, value in dict(eval_metrics).items():
    #    tf.summary.scalar(name, value, step=round_num)

    #print('round {:2d}, metrics={}'.format(round_num, metrics))


state = iterative_process.initialize()

for round_num in range(1, NUM_ROUNDS):

    clients_set = random.choices(emnist_train.client_ids, k=random.randint(1,10))
    print("round {} clients:{}".format(round_num, clients_set))
    federated_train_data = make_federated_data(emnist_train, clients_set)
    federated_test_data = make_federated_data(emnist_test, emnist_train.client_ids[0:50])

    state, metrics = iterative_process.next(state, federated_train_data)

    eval_metrics = evaluation(state.model, federated_test_data)
    print('round {:2d}, eval_metrics={}'.format(round_num, eval_metrics))

    y2.append(eval_metrics['loss'])
    Y2.append(eval_metrics['sparse_categorical_accuracy'])

    #for name, value in dict(eval_metrics).items():
    #    tf.summary.scalar(name, value, step=round_num)

    #print('round {:2d}, metrics={}'.format(round_num, metrics))



plt.title('Loss')
plt.xlabel('round number')
plt.ylabel('loss')
plt.plot(x,y1,label='10 clients')
plt.plot(x,y2, label='2 clients')
plt.legend()
plt.show()

plt.title('Acc')
plt.xlabel('round number')
plt.ylabel('Acc')
plt.plot(x,Y1,label='10 clients')
plt.plot(x,Y2, label='2 clients')
plt.legend()
plt.show()
'''
