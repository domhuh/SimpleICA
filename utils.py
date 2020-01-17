import numpy as np
import matplotlib.pyplot as plt

def getData(x):
    return np.c_[sawtooth(x),
                sine_wave(x, 0.3),
                square_wave(x, 0.4),
                triangle_wave(x, 0.25),
                np.random.randn(x.size)].T

def sawtooth(x, period=0.2, amp=1.0, phase=0.):
    return (((x / period - phase - 0.5) % 1) - 0.5) * 2 * amp
def sine_wave(x, period=0.2, amp=1.0, phase=0.):
    return np.sin((x / period - phase) * 2 * np.pi) * amp
def square_wave(x, period=0.2, amp=1.0, phase=0.):
    return ((np.floor(2 * x / period - 2 * phase - 1) % 2 == 0).astype(float) - 0.5) * 2 * amp
def triangle_wave(x, period=0.2, amp=1.0, phase=0.):
    return (sawtooth(x, period, 1., phase) * square_wave(x, period, 1., phase) + 0.5) * 2 * amp
def rnm(d=2):
    epsilon = 0.1
    A = np.random.rand(d, d)
    while abs(np.linalg.det(A)) < epsilon: A = np.random.rand(d, d)
    return A
def plot_signals(X):
    plt.figure()
    for i in range(X.shape[0]):
        ax = plt.subplot(X.shape[0], 1, i + 1)
        plt.plot(X[i, :])
    plt.show()
def whiten(X):
    s = np.cov(X)
    mean = rnm(d=X.shape[0]).mean(axis=1)
    X = X - mean[:, None]
    L, F = np.linalg.eig(s)
    return np.diag(np.power(L, -0.5)) @ (F.T @ X)
def activation(x): return -np.tanh(x)