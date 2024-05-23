import numpy as np
import jax.numpy as jnp
from jax import Array
from scipy.optimize import minimize
import jax


def to_cartesian(angles: np.ndarray) -> np.ndarray:
    """
    angles: all between [0,\pi) except last one between [0,2\pi)
    """
    # https://stackoverflow.com/a/20133681
    dummy = np.array([2 * np.pi])
    angles = np.concatenate((dummy, angles))
    sines = np.sin(angles)
    sines[0] = 1
    prods = np.cumprod(sines)
    cosines = np.cos(angles)
    cosines = np.roll(cosines, -1)
    return prods * cosines


def to_cartesian_jax(angles: np.ndarray) -> Array:
    """
    angles: all between [0,\pi) except last one between [0,2\pi)
    """
    # https://stackoverflow.com/a/20133681
    dummy = jnp.array([2 * np.pi])
    angles = jnp.concatenate((dummy, angles))
    sines = jnp.sin(angles)
    sines = sines.at[0].set(1)
    prods = jnp.cumprod(sines)
    cosines = jnp.cos(angles)
    cosines = jnp.roll(cosines, -1)
    return prods * cosines


def predict(angles: np.ndarray, X: np.ndarray) -> np.ndarray:
    coeffs = to_cartesian(angles)
    predictions = np.dot(X, coeffs)
    return predictions


def predict_jax(angles: np.ndarray, X: np.ndarray) -> Array:
    coeffs = to_cartesian_jax(angles)
    predictions = jnp.dot(X, coeffs)
    return predictions


def angular_coordinates(x: np.ndarray) -> np.ndarray:
    # https://stackoverflow.com/a/77442105
    a = x[1:] ** 2
    b = np.sqrt(np.cumsum(a[::-1], axis=0))
    phi = np.arctan2(b[::-1], x[:-1])
    phi[-1] *= np.sign(x[-1])
    return phi


class UnitSphereRegression:
    def __init__(self):
        self.angles = None

    @property
    def coeff_(self):
        return to_cartesian(self.angles)

    def fit(self, X, y):
        def loss_function_jax(angles):
            predictions = predict_jax(angles, X)
            return jnp.mean((predictions - y) ** 2)

        grad_func = jax.grad(loss_function_jax)
        hessian_func = jax.jacfwd(jax.grad(loss_function_jax))

        n = X.shape[1] - 1
        interval_lens = np.ones(n) * np.pi / 2

        initial_angles = interval_lens / 2

        result = minimize(
            loss_function_jax,
            initial_angles,
            jac=grad_func,
            hess=hessian_func,
            method='trust-constr',
        )

        self.angles = result.x
        self.angles = np.clip(self.angles, 0, interval_lens)
        # Rather than adding bounds to minimize we clip
        # the results, because bounds worsen the quality
        # of the result

        return self

    def predict(self, X):
        return np.dot(X, self.coeff_)
