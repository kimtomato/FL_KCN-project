import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

vps = 10

parser = argparse.ArgumentParser(
    description='Analyze and plot results of FL/VKN simulations.')
parser.add_argument(
    '-i',
    '--input_files',
    dest='inputs',
    nargs='+',
    help='<Required> List of input simulation results pickle files to consider (useful for exactly the same type of simulations with different seeds)',
    required=True)
args = parser.parse_args()

 #Example input:
 #-i optimized_rpgm_dump10_0_10 optimized_rpgm_dump10_10_21 optimized_rpgm_dump10_21_25



stats = {'vkn': {vps: []}, 'tradi': {vps: []}}

for input_file in args.inputs:

    with open(input_file, 'rb') as filehandler:
        stats_local = pickle.load(filehandler)

    # Add vkn stats
    for st in stats_local['vkn'][vps]:
        stats['vkn'][vps].append(st)

    # Add traditional stats
    for st in stats_local['tradi'][vps]:
        stats['tradi'][vps].append(st)


print(stats.keys())

stats_vkn = stats['vkn'][vps]
stats_tradi = stats['tradi'][vps]


# vkn curve
def preprocess_stat(stat, threshold):
    for i in range(0, threshold):
        if i in stat.training_accuracy:
            stat.training_accuracy.pop(i)


def preprocess_stats(stats_data):
    max_minvalue = None
    for stat in stats_data:
        if max_minvalue is None or max_minvalue < min(
                list(stat.training_accuracy.keys())):
            max_minvalue = min(list(stat.training_accuracy.keys()))

    for stat in stats_data:
        preprocess_stat(stat, max_minvalue)
        #print (max(list(stat.training_accuracy.keys())))


preprocess_stats(stats_vkn)
preprocess_stats(stats_tradi)


def get_average_nbselect(stats, smooth):
    (x, _) = stats[0].get_step_nbselect(smooth)
    y = []
    error = []

    for (i, stepid) in enumerate(x):

        vals = []
        for stat in stats:
            vals.append(stat.get_step_nbselect(smooth)[1][i])

        vals = np.array(vals)

        y.append(np.mean(vals))
        # Student law for 25 samples
        error.append(1.708 * (np.std(vals) / np.sqrt(len(vals))))

    x = np.array(x)
    y = np.array(y)
    error = np.array(error)


    return (x, y, error)


def get_average_loss(stats, smooth):
    (x, _) = stats[0].get_training_accuracy(smooth)
    y = []
    error = []

    for (i, stepid) in enumerate(x):

        vals = []
        for stat in stats:
            vals.append(stat.get_training_loss(smooth)[1][i])

        vals = np.array(vals)

        y.append(np.mean(vals))
        error.append(1.708 * (np.std(vals) / np.sqrt(len(vals))))

    x = np.array(x)
    y = np.array(y)
    error = np.array(error)
    print(x, y)

    return (x, y, error)


def get_average_acc(stats, smooth):
    (x, _) = stats[0].get_training_accuracy(smooth)
    y = []
    error = []

    for (i, stepid) in enumerate(x):

        vals = []
        for stat in stats:
            vals.append(stat.get_training_accuracy(smooth)[1][i])

        vals = np.array(vals)

        y.append(np.mean(vals))
        # Student law for 25 samples
        error.append(1.708 * (np.std(vals) / np.sqrt(len(vals))))

    x = np.array(x)
    y = np.array(y)
    error = np.array(error)
    print(x,y)

    return (x, y, error)


def get_average_eff(stats, smooth):
    (x, _) = stats[0].get_step_efficiency(smooth)
    y = []
    error = []

    for (i, stepid) in enumerate(x):

        vals = []
        for stat in stats:
            vals.append(stat.get_step_efficiency(smooth)[1][i])

        vals = np.array(vals)

        y.append(np.mean(vals))
        # Student law for 25 samples
        error.append(1.708 * (np.std(vals) / np.sqrt(len(vals))))

    x = np.array(x[25 // smooth:])
    y = np.array(y[25 // smooth:])
    error = np.array(error[25 // smooth:])

    return (x, y, error)


def plot_acc_comparison(vkn, tradi):
    (x, y, error) = get_average_acc(stats_vkn, 4)
    (xtrad, ytrad, error_trad) = get_average_acc(stats_tradi, 4)

    plt.title('Model Accuracy per training step')

    plt.fill_between(x, y - error, y + error, color='blue', alpha=0.3)
    plt.plot(
        x,
        y,
        color='blue',
        label="VKN-orchestrated training",
        linewidth=0.8)

    plt.fill_between(
        xtrad,
        ytrad -
        error_trad,
        ytrad +
        error_trad,
        color='orange',
        alpha=0.5)
    plt.plot(
        xtrad,
        ytrad,
        color='red',
        label="Traditional training",
        linewidth=0.8)

    plt.xlabel("Training step")
    plt.ylabel("Model accuracy ∈ [0;1]")
    plt.legend()

    plt.show()


def plot_loss_comparison(vkn, tradi):
    (x, y, error) = get_average_loss(stats_vkn, 4)
    (xtrad, ytrad, error_trad) = get_average_loss(stats_tradi, 4)

    plt.title('Loss per training step')

    plt.fill_between(x, y - error, y + error, color='blue', alpha=0.3)
    plt.plot(
        x,
        y,
        color='blue',
        label="VKN-orchestrated training",
        linewidth=0.8)

    plt.fill_between(
        xtrad,
        ytrad -
        error_trad,
        ytrad +
        error_trad,
        color='orange',
        alpha=0.5)
    plt.plot(
        xtrad,
        ytrad,
        color='red',
        label="Traditional training",
        linewidth=0.8)

    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()


def plot_nbselect_comparison(vkn, tradi):
    (x, y, error) = get_average_nbselect(stats_vkn, 10)

    xtrad = np.copy(x)
    ytrad = []
    error_trad = []
    for _ in xtrad:
        ytrad.append(10)
        error_trad.append(0)

    ytrad = np.array(ytrad)
    error_trad = np.array(error_trad)

    #(xtrad,ytrad,error_trad) = get_average_nbselect(stats_tradi,10)

    plt.title('nbselect')
    plt.fill_between(x, y - error, y + error, color='blue')
    plt.plot(x, y, color='purple')

    plt.fill_between(
        xtrad,
        ytrad - error_trad,
        ytrad + error_trad,
        color='orange')
    plt.plot(xtrad, ytrad, color='red')
    plt.show()


def plot_eff_comparison(vkn, tradi):
    (x, y, error) = get_average_eff(stats_vkn, 4)
    (xtrad, ytrad, error_trad) = get_average_eff(stats_tradi, 4)

    plt.fill_between(x, y - error, y + error, color='blue')
    plt.plot(x, y, color='purple')

    plt.fill_between(
        xtrad,
        ytrad - error_trad,
        ytrad + error_trad,
        color='orange')
    plt.plot(xtrad, ytrad, color='red')
    plt.show()


def plot_acc_nbselect():
    (x, y, error) = get_average_acc(stats_vkn, 4)
    (xtrad, ytrad, error_trad) = get_average_acc(stats_tradi, 4)

    fig, ax1 = plt.subplots()
    ax1.set_title("Model Accuracy per training step")

    color = 'black'
    ax1.set_xlabel('Training step')
    ax1.set_ylabel('Model accuracy ∈ [0;1]', color=color)

    ax1.fill_between(x, y - error, y + error, color='blue', alpha=0.3)
    ax1.plot(x, y, color='blue', linewidth=0.8,
             label='VKN-orchestrated training - Accuracy')

    ax1.fill_between(
        xtrad,
        ytrad -
        error_trad,
        ytrad +
        error_trad,
        color='orange',
        alpha=0.3)
    ax1.plot(
        xtrad,
        ytrad,
        color='red',
        label="Traditional training - Accuracy",
        linewidth=0.8)

    ax1.set_ylim(0.16, 0.95)
    ax1.legend(loc=2)

    ax1.tick_params(axis='y', labelcolor=color)

    '''
    plt.title('Model Accuracy per training step')

    plt.fill_between(x, y-error, y+error, color='blue', alpha=0.3)
    plt.plot(x,y,color='blue', label="VKN-orchestrated training - Accuracy", linewidth=0.8)

    plt.fill_between(xtrad, ytrad-error_trad, ytrad+error_trad, color='orange', alpha=0.5)
    plt.plot(xtrad,ytrad, color='red', label="Traditional training - Accuracy", linewidth=0.8)
    '''

    (x, y, error) = get_average_nbselect(stats_vkn, 10)

    xtrad = np.copy(x)
    ytrad = []
    error_trad = []
    for _ in xtrad:
        ytrad.append(10)
        error_trad.append(0)

    ytrad = np.array(ytrad)
    error_trad = np.array(error_trad)

    matplotlib.rcParams['hatch.linewidth'] = 10.0

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylim(7, 15)
    color = 'black'
    ax2.set_ylabel('Number of selected training nodes per step',
                   color=color)  # we already handled the x-label with ax1

    ax2.fill_between(
        x,
        y,
        ytrad,
        label='Bandwidth saved through VKN orchestration',
        alpha=0.1,
        hatch='/',
        color='gray',
        edgecolor='white')

    ax2.fill_between(x, y - error, y + error, color='blue', alpha=0.3)
    ax2.plot(
        x,
        y,
        color='blue',
        linestyle='dashed',
        label='VKN-orchestrated training - Selected nodes',
        linewidth=0.8)

    ax2.plot(
        xtrad,
        ytrad,
        color='red',
        linestyle='dashed',
        label='Traditional training - Selected nodes',
        linewidth=0.8)

    ax2.legend(loc=4)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    '''
    plt.fill_between(x, y-error, y+error, color='blue', alpha=0.3)
    plt.plot(x,y,color='blue', label="VKN-orchestrated training - Nb select", linewidth=0.8)

    plt.fill_between(xtrad, ytrad-error_trad, ytrad+error_trad, color='orange', alpha=0.5)
    plt.plot(xtrad,ytrad, color='red', label="Traditional training - Nb select", linewidth=0.8)


    plt.xlabel("Training step")
    plt.ylabel("Model accuracy ∈ [0;1]")
    plt.legend()

    plt.show()'''


# plot_acc_nbselect() RPGM optimizepr
plot_acc_comparison(stats_vkn, stats_tradi)
plot_loss_comparison(stats_vkn, stats_tradi)

#plot_nbselect_comparison(stats_vkn, stats_tradi)
#plot_eff_comparison(stats_vkn, stats_tradi)
