from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np


class SimpleRegressor:
    def __init__(self):
        raise NotImplementedError

    def predict(self, x):
        return self.weights_[0] + x * self.weights_[1]

    def fit(self, x, y, lr: float, steps: int = 20) -> None:
        # x and y are vectors of the same length

        self.weights_ = np.zeros(2)  # a vector with the two parameters of our model

        # we keep copies of the x, y and the predicted outputs at each
        # iteration to use them in show_training() later
        self.x_train_ = x.copy()
        self.y_train_ = y.copy()
        self.y_predicted_ = np.zeros((steps + 1, y.shape[0]))
        self.y_predicted_[0] = self.predict(x)

        for i in range(steps):
            # here: update self.weights_?
            self.y_predicted_[i + 1] = self.predict(x)

        raise NotImplementedError  # you should override this method!

    def show_training(self):
        show_regression_training_(self.x_train_, self.y_train_, self.y_predicted_)


def show_regression_training_(x_train, y_train, y_predicted):
    fig, (loss_ax, graph_ax) = plt.subplots(1, 2, layout="tight")
    fig.set_size_inches(12, 5)

    # MSE loss
    loss = np.pow(y_predicted - y_train, 2).mean(axis=-1)
    loss_ax.set(title="MSE loss", ylabel="loss", xlabel="iteration", ylim=(0, loss.max() * 1.1))
    loss_ax.set_box_aspect(1)
    loss_ax.plot(loss)
    loss_ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Intermediate models
    steps = y_predicted.shape[0]
    idx = np.linspace(0, steps - 1, num=min(steps, 500), dtype=int)
    order = np.argsort(x_train)
    x_sorted = x_train[order]
    y_subset = y_predicted[idx][:, order]  # (n_lines, n_points)

    segments = np.stack([np.broadcast_to(x_sorted, y_subset.shape), y_subset], axis=-1)
    cmap = plt.colormaps["viridis"]
    norm = Normalize(vmin=0, vmax=steps - 1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, array=idx, label="model")
    graph_ax.add_collection(lc)
    graph_ax.set(title="Intermediate models")
    graph_ax.scatter(x_train, y_train, label="data", zorder=5)
    graph_ax.autoscale()

    plt.colorbar(lc, ax=graph_ax, label="iteration")
    plt.legend()
    plt.show(fig)
