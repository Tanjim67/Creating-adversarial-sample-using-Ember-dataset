import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator



def plot_norm(x1, x2, y1, y2, xlabel, legend):
    """
    Generating plots for the 4 different norms
    :param x1: array for x values first line. Will be the same on all plots
    :param x2: array for x values second line. Will be the same on all plots
    :param y1: array of length four, one for each plot. Each array contains y values for for first line.
    :param y2: array of length four, one for each plot. Each array contains y values for for second line.
    :param xlabel: lable for x axis. Y axis will always be l0, l1, l2, linf norm. One for each plot.
    :param legend: array with two strings for legend, one for each line. Will be the same on all plots.
    :return: None
    """
    plt.figure(figsize=(13, 10))
    plt.subplot(221)
    plt.plot(x1, y1[0], "*-", label=legend[0])
    plt.plot(x2, y2[0], "*-", label=legend[1])
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel('L0 norm')
    plt.xlabel(xlabel)
    plt.title("Norm vs Epsilon")
    plt.legend()

    plt.subplot(222)
    plt.plot(x1, y1[1], "*-", label=legend[0])
    plt.plot(x2, y2[1], "*-", label=legend[1])
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel('L1 norm')
    plt.xlabel(xlabel)
    plt.title("Norm vs Epsilon")
    plt.legend()

    plt.subplot(223)
    plt.plot(x1, y1[2], "*-", label=legend[0])
    plt.plot(x2, y2[2], "*-", label=legend[1])
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel('L2 norm')
    plt.xlabel(xlabel)
    plt.legend()

    plt.subplot(224)
    plt.plot(x1, y1[3], "*-", label=legend[0])
    plt.plot(x2, y2[3], "*-", label=legend[1])
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel('Linf norm')
    plt.xlabel(xlabel)
    plt.legend()

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

def plot_hist(hist):
    epochs, t_loss, t_acc, v_loss, v_acc = zip(*hist)
    plt.figure(figsize=(16, 6))
    plt.plot([1, 2])
    plt.subplot(121)
    plt.plot(epochs, t_loss, label="train loss")
    plt.plot(epochs, v_loss, label="validation loss")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('Model Loss')
    plt.legend()
    plt.subplot(122)
    plt.plot(epochs, t_acc, label="training accuracy")
    plt.plot(epochs, v_acc, label="validation accuracy")
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title('Model Accuracy')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()


def plot_eps_acc(epsilons, accuracies_mask, accuracies_no_mask, accuracies_mal):
    plt.figure(figsize=(5, 5))
    plt.plot(epsilons, accuracies_mask, "*-")
    plt.plot(epsilons, accuracies_no_mask, "*-")
    plt.plot(epsilons, accuracies_mal, "*-")
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.legend(['constraints', 'no constraints', 'only malicous'], loc='upper right')
    plt.show()
