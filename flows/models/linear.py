from typing import List, Optional, Union
from jax import config
from jax import numpy as jnp
from flax import linen as nn
from numpy.typing import NDArray
import numpy as np
import jax
# from ..types import evaluationMode # Assuming evaluationMode is an Enum with members 'direct' and 'inverse'

config.update("jax_enable_x64", True)


def _cond_b_infinite(b, interval):
    return b


def _cond_b_open_right(b, interval):
    return b**2 + interval[0]


def _cond_b_finite(b, interval):
    left = interval[0]
    len_interval = interval[1] - interval[0]
    effective_b = jax.nn.sigmoid(b) * len_interval
    return left + effective_b


def _cond_a_infinite(a, b, interval):
    return a


def _cond_a_open_right(a, b, interval):
    return a**2


def _cond_a_finite(a, b, interval):
    original_len = 2.0
    new_len_max = interval[1] - b
    effective_a = jax.nn.sigmoid(a)
    return effective_a / original_len * new_len_max



class Linear(nn.Module):
    """
    Implements an element-wise affine transformation, $y = a \odot x + b$, 
    which is an **invertible** operation.

    This module is often used as a simple scaling and shifting layer in deep learning 
    or as a coupling layer component in Normalizing Flows.

    The parameters $a$ (scale) and $b$ (shift) can optionally be made **learnable** Flax parameters using `opt_a` and `opt_b`.

    Attributes:
        a (Union[List[float], NDArray]): The scaling factor(s). Must have a shape 
                                         compatible with the input $x$ for element-wise multiplication.
        b (Union[List[float], NDArray]): The shifting factor(s). Must have a shape 
                                         compatible with the input $x$ for element-wise addition.
        opt_a (Optional[bool]): If True, 'a' is registered as a learnable Flax parameter. 
                                Defaults to True.
        opt_b (Optional[bool]): If True, 'b' is registered as a learnable Flax parameter. 
                                Defaults to True.
    """

    a: Union[List[float], NDArray[np.float64]]
    b: Union[List[float], NDArray[np.float64]]
    opt_a: Optional[bool] = True
    opt_b: Optional[bool] = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, mode='direct') -> jnp.ndarray:
        """
        Applies the affine transformation or its inverse.

        Args:
            x (jnp.ndarray): The input data.
            mode (evaluationMode, optional): The evaluation mode. Must be 'direct' 
                                            (forward) or 'inverse' (backward). Defaults to 'direct'.

        Returns:
            jnp.ndarray: The output of the transformation.
        """
        # --- Parameter Initialization/Loading ---
        if self.opt_a:
            # Register 'a' as a learnable parameter
            a = self.param("linear_a", lambda *_: jnp.asarray(self.a), jnp.shape(self.a))
        else:
            # Treat 'a' as a static constant
            a = jnp.asarray(self.a)
            
        if self.opt_b:
            # Register 'b' as a learnable parameter
            b = self.param("linear_b", lambda *_: jnp.asarray(self.b), jnp.shape(self.b))
        else:
            # Treat 'b' as a static constant
            b = jnp.asarray(self.b)
        
        # --- Transformation Logic ---
        if mode == 'inverse':
            # Inverse: $x = (y - b) / a$
            return (x - b) / a
        else:
            # Direct: $y = x \odot a + b$
            return x * a + b
        

class LinearOnInterval(nn.Module):
    """
    Trainable linear transformation a * x + b.
    In each dimension, maps [-1,1] to the interval specified
    in intervals.
    """

    a: Union[List[float], NDArray[np.float64]]
    b: Union[List[float], NDArray[np.float64]]
    intervals: NDArray[np.float64]
    opt_a: Optional[bool] = True
    opt_b: Optional[bool] = True

    def setup(self):
        # if jnp.any(self.intervals[:, 1] - self.intervals[:, 0] <= 0.0):
        #     raise AttributeError("The input intervals are not strictly ascending.")

        def _apply_cond_b(index, b, intervals):
            return jax.lax.switch(
                index,
                [_cond_b_infinite, _cond_b_open_right, _cond_b_finite],
                b,
                intervals,
            )

        def _apply_cond_a(index, a, b, intervals):
            return jax.lax.switch(
                index,
                [_cond_a_infinite, _cond_a_open_right, _cond_a_finite],
                a,
                b,
                intervals,
            )

        self.index = jnp.where(
            (self.intervals[:, 0] == -jnp.inf) & (self.intervals[:, 1] == jnp.inf),
            0,
            jnp.where(
                (self.intervals[:, 0] != -jnp.inf) & (self.intervals[:, 1] == jnp.inf),
                1,
                2,
            ),
        )
        
        self.cond_b = lambda b: jax.vmap(_apply_cond_b, (0, 0, 0))(
            self.index, b, self.intervals
        )
        self.cond_a = lambda a, b: jax.vmap(_apply_cond_a, (0, 0, 0, 0))(
            self.index, a, b, self.intervals
        )
        
        ## Take out possible zeros in b that for [l, inf) leads to non-optimization of parameter
        self.b2 = jnp.where((self.index == 1) & (jnp.asarray(self.b) == 0.), jnp.asarray(self.b)+ 1e-3, jnp.asarray(self.b)  )

    @nn.compact
    def __call__(self, x, inverse: bool = False):
        if self.opt_b:
            b = self.param(
                "linear_b", lambda *_: jnp.asarray(self.b2), jnp.shape(self.b2)
            )
        else:
            b = jnp.asarray(self.b)

        if self.opt_a:
            a = self.param(
                "linear_a", lambda *_: jnp.asarray(self.a), jnp.shape(self.a)
            )
        else:
            a = jnp.asarray(self.a)

        b = self.cond_b(b)
        a = self.cond_a(a, b)

        if inverse:
            return a * (x + 1) + b
        return (x - b) / a - 1