import numpy
import sys

def activation_function(Z, name):

    if name == "sigmoid":
        return 1.0 / (1.0 + numpy.exp(-Z))

    if name == "tanh":
        return numpy.tanh(Z)

    if name == "relu":
        return numpy.maximum(0, Z)

    if name == "leaky_relu":
        return numpy.maximum(0.01 * Z, Z)


def derivative_activation_function(A, name):
    if name == "sigmoid":
        return A * (1 - A)

    if name == "tanh":
        return 1 - A * A

    if name == "relu":
        return (A > 0) * 1.0

    if name == "leaky_relu":
        return (A > 0) * 1.0 + (A < 0) * 0.01


def show_progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + ' ' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()