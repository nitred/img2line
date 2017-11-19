"""Lonely file consisting of the simple implementation of img2line."""
import numpy as np
from skimage.io import imread
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def get_line_from_image(imgpath, plot=False):
    """Function to return the x and y coordinates of the line in the image.

    Args:
        imgpath (string): Full path to an image of a line.
        plot (bool): Whether to plot the results or not.

    Returns:
        x, y (nd.array, ndarray): The coordinates of the lines. The coordinates
            start from 0,0 i.e. (x[0], y[0]) = (0, 0).
    """
    # Read image
    imgarr = imread(imgpath, as_grey=True)
    # Threshold image.
    imgarr[imgarr < 0.5] = 0
    imgarr[imgarr >= 0.5] = 1
    # Flip image to make covert to more intuitive cartesian coordinates.
    imgarrflip = imgarr[::-1]  # bottom left = 0,0
    colmin = np.min(imgarrflip, axis=0)  # minimum values of every column
    x_min = np.argmin(colmin)  # searching for first zero from the start
    x_max = len(colmin) - np.argmin(colmin[::-1], axis=0) - 1  # searching for first zero from the last
    x = np.arange(x_min, x_max + 1)  # + 1 to include x_max as well
    y = np.argmin(imgarrflip[:, x], axis=0)  # y coordinates from the slice determined by x coordinates
    # Convert x and y to have minimum value as zero.
    x = x - np.min(x)
    y = y - np.min(y)

    if plot is True:
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(x, y)
        plt.xlim(0, len(x))
        plt.ylim(min(y), max(y))
        plt.show()

    return x, y


def plot_line_and_model(x, y, model, label):
    """Plot original line and model prediction."""
    from matplotlib import pyplot as plt
    # to keep the order
    x_orig = np.copy(x)
    y_orig = np.copy(y)

    X_test = x_orig[:, np.newaxis]
    # Y_test = y_orig

    plt.figure()
    plt.scatter(x_orig, y_orig,
                color='navy',
                s=30,
                marker='o',
                label="training points")
    Y_pred = model.predict(X_test)
    plt.plot(X_test, Y_pred, color='orange', linewidth=2, label=label)
    plt.show()


def fit_line_to_image(imgpath, degree=5, plot=False):
    """Function to fit a line to the image of a line.

    Args:
        imgpath (string): Full path to an image of a line.
        degree (int): Degree of polynomial to line fit.
        plot (bool): Whether to plot the results or not.

    Returns:
        x, y, model (nd.array, nd.array, sklearn.pipeline):
        - x, y are the coordinates of the lines. The coordinates start from 0,0
            i.e. (x[0], y[0]) = (0, 0).
        - model can be used to predict y-coordinates from other x-coordinates by
            calling `y_new = model.predict(x_new)`.
    """
    x, y = get_line_from_image(imgpath, plot=False)

    # shuffle x and y
    random_order = np.random.permutation(len(x))
    X_train = x[random_order][:, np.newaxis]
    Y_train = y[random_order]

    # Train model
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X_train, Y_train)

    if plot is True:
        plot_line_and_model(x, y, model, label="degree {}".format(degree))

    return x, y, model


def fit_gpr_to_image(imgpath, plot=False):
    """Function to fit a GaussianProcessRegressor to the image of a line.

    Args:
        imgpath (string): Full path to an image of a line.
        plot (bool): Whether to plot the results or not.

    Returns:
        x, y, model (nd.array, nd.array, sklearn.GaussianProcessRegressor):
        - x, y are the coordinates of the lines. The coordinates start from 0,0
            i.e. (x[0], y[0]) = (0, 0).
        - model can be used to predict y-coordinates from other x-coordinates by
            calling `y_new = model.predict(x_new)`.
    """
    x, y = get_line_from_image(imgpath, plot=False)

    # shuffle x and y
    random_order = np.random.permutation(len(x))
    X_train = x[random_order][:, np.newaxis]
    Y_train = y[random_order]

    # Train Model
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    model.fit(X_train, Y_train)

    if plot is True:
        plot_line_and_model(x, y, model, label="GPR")

    return x, y, model
