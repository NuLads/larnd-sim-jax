"""Display curve for the a06 loss panels: a cubic Hermite through every real loss value, with the real jax.grad slope wherever a point has one and a
monotone (PCHIP) slope at value-only points. Because the curve uses the gradients, a tangent lying on it proves nothing: the independent test of the
gradient is the finite-difference comparison. audit() lists any interval where the drawn curve leaves the range of its end values."""
import numpy as np
from scipy.interpolate import PchipInterpolator


class Curve:
    def __init__(self, wn, Ln, wg, dW):
        self.w, self.y = np.asarray(wn, float), np.asarray(Ln, float)
        self.has = np.array([np.min(np.abs(np.asarray(wg) - x)) < 1e-9 * max(1, abs(x)) for x in self.w])
        d = PchipInterpolator(self.w, self.y).derivative()(self.w)
        for x, g in zip(wg, dW): d[int(np.argmin(np.abs(self.w - x)))] = g
        self.d = d; self.h = np.diff(self.w); self.D = np.diff(self.y) / self.h

    def __call__(self, x):
        x = np.atleast_1d(np.asarray(x, float)); i = np.clip(np.searchsorted(self.w, x) - 1, 0, len(self.h) - 1); h = self.h[i]; s = (x - self.w[i]) / h
        return ((2 * s ** 3 - 3 * s ** 2 + 1) * self.y[i] + (s ** 3 - 2 * s ** 2 + s) * h * self.d[i] + (-2 * s ** 3 + 3 * s ** 2) * self.y[i + 1] + (s ** 3 - s ** 2) * h * self.d[i + 1])


def audit(c, f):
    rows = []
    for i in range(len(c.h)):
        ys = c(np.linspace(c.w[i], c.w[i + 1], 801)); lo, hi = min(c.y[i], c.y[i + 1]), max(c.y[i], c.y[i + 1]); over = max(ys.max() - hi, lo - ys.min(), 0.0)
        rows.append((round(float(f(c.w[i])), 4), round(float(f(c.w[i + 1])), 4), round(float(over), 2), bool(c.d[i] * c.d[i + 1] < 0)))
    return rows
