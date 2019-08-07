import matplotlib.pyplot as plt
import os
import gridsearch as gs

def plot(filename, precision, recall, Fscore, x_values):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    plt.plot(x_values, precision, label='precision')
    plt.plot(x_values, recall, label='recall')
    plt.plot(x_values, Fscore, label="F_{} score".format(gs.Fbeta))
    plt.ylim((0,1))
    plt.grid(True)
    plt.title(os.path.splitext(os.path.basename(filename))[0])
    plt.legend()
    plt.savefig(filename)
    plt.clf()
