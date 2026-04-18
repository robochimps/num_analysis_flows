from typing import Callable, List, Optional, Union
from jax import config
from jax import numpy as jnp
from flax import linen as nn
from numpy.typing import NDArray
import flows 
from .MLP import NormalizedMultiLayerPerceptron
config.update("jax_enable_x64", True)
from ..types import *
from .linear import Linear, LinearOnInterval
from ..utils import lipswish, sinh, cosh, _erf, scaled_erf
import jax 
import numpy as np

class ActivationFunction(Enum):
    """Special types of activation functions for invertible MLP mappings"""

    LIPSWISH = lipswish
    RELU = nn.relu
    IDENTITY = lambda x: x


class _InvertibleDenseBlockFourier(nn.Module):
    features: List[int]
    activations: List[activationType]
    lipschitz_constant: Optional[float] = 0.9

    @nn.compact
    def __call__(self, x):
        size_ = x.shape[-1]
        for i, size in enumerate(self.features):
            kernel = self.param(
                f"w_{i}", jax.nn.initializers.glorot_uniform(), (size_, size)
            )
            sv = jnp.max(_singular_values_fourier(kernel, jnp.shape(kernel)))
            kernel = jax.lax.cond(
                sv >= self.lipschitz_constant,
                lambda a, b: self.lipschitz_constant * a / b,
                lambda a, b: a,
                kernel,
                sv,
            )
            bias = self.param(f"b_{i}", jax.nn.initializers.zeros, (size,))
            x = self.activations[i](jnp.dot(x, kernel) + bias)
            size_ = size
        return x

def _singular_values_fourier(kernel, input_shape):
    transforms = jnp.fft.fft2(kernel, input_shape, axes=[0, 1])
    return jnp.linalg.svd(transforms, compute_uv=False)


def _singular_values(kernel, input_shape):
    return jnp.linalg.svd(kernel, compute_uv=False)


class Tanh(nn.Module):
    """Tanh and its inverse"""


    @nn.compact
    def __call__(self, x, inverse: bool = False):
        if inverse:
            return jnp.arctanh(x)
        else:
            return jnp.tanh(x)



class _Inverse(nn.Module):
    @nn.compact
    def __call__(self, x, k, dense):
        xk, x0 = x
        x = [x0 - dense(xk), x0]
        return x, x
    
class _Inverse_Halpern(nn.Module):
    @nn.compact
    def __call__(self, x, k, dense):
        xk, x0 = x
        Tx = x0 - dense(xk)
        x = [1/(k+2) * x0 + (1 - 1/(k+2)) * Tx, x0]
        return x, x
    
class _Inverse_Picardi(nn.Module):
    @nn.compact
    def __call__(self, x, k, dense):
        xk, x0 = x
        Tx = x0 - dense(xk)
        x = [1/(k+2) * xk + (1 - 1/(k+2)) * Tx, x0]
        return x, x

class _InvertibleDenseBlockSVD_mult(nn.Module):
    features: List[int]
    activations: List[activationType]
    lipschitz_constant: Optional[float] = 0.9

    @nn.compact
    def __call__(self, x):
        size_ = x.shape[-1]
        for i, size in enumerate(self.features):
            kernel = self.param(
                f"w_{i}", jax.nn.initializers.glorot_uniform(), (size_, size)
            )
            bias = self.param(f"b_{i}", jax.nn.initializers.zeros, (size,))
            L, S, R = jnp.linalg.svd(kernel, full_matrices=False)

            def update_S(x):
                return jax.lax.cond(
                    x >= self.lipschitz_constant,
                    lambda _: self.lipschitz_constant,
                    lambda _: x,
                    None,
                )

            S_updated = jax.vmap(update_S)(S)
            kernel = jnp.dot(L * S_updated, R)
            x = self.activations[i](jnp.dot(x, kernel) + bias)
            size_ = size
        return x


class _InvertibleDenseBlockSVD(nn.Module):
    features: List[int]
    activations: List[activationType]
    lipschitz_constant: Optional[float] = 0.9

    @nn.compact
    def __call__(self, x):
        size_ = x.shape[-1]
        for i, size in enumerate(self.features):
            kernel = self.param(
                f"w_{i}", jax.nn.initializers.glorot_uniform(), (size_, size)
            )
            sv = jnp.max(_singular_values(kernel, jnp.shape(kernel)))
            switch = jnp.heaviside(self.lipschitz_constant, sv)
            kernel = switch * kernel /sv * self.lipschitz_constant + \
                     ( 1 - switch) * kernel
            bias = self.param(f"b_{i}", jax.nn.initializers.zeros, (size,))
            x = self.activations[i](jnp.dot(x, kernel) + bias)
            size_ = size
        return x


class _InvertibleDenseBlockSVD_mult(nn.Module):
    features: List[int]
    activations: List[ActivationFunction]
    lipschitz_constant: Optional[float] = 0.9

    @nn.compact
    def __call__(self, x):
        size_ = x.shape[-1]
        for i, size in enumerate(self.features):
            kernel = self.param(
                f"w_{i}", jax.nn.initializers.glorot_uniform(), (size_, size)
            )
            bias = self.param(f"b_{i}", jax.nn.initializers.zeros, (size,))
            L, S, R = jnp.linalg.svd(kernel, full_matrices=False)

            def update_S(x):
                return jax.lax.cond(
                    x >= self.lipschitz_constant,
                    lambda _: self.lipschitz_constant,
                    lambda _: x,
                    None,
                )

            S_updated = jax.vmap(update_S)(S)
            kernel = jnp.dot(L * S_updated, R)
            x = self.activations[i](jnp.dot(x, kernel) + bias)
            size_ = size
        return x



class Inverse(nn.Module):
    """
    A small helper module designed for use within `jax.nn.scan` to perform 
    a single iteration of the fixed-point iteration (e.g., Picard iteration) 
    required for numerical inversion of a ResNet block:

    $$x_{k+1} = x_0 - f(x_k)$$
    
    where $x_0$ is the output of the ResNet block, and $f$ is the non-linear transformation 
    (the dense network).

    This module is stateless and intended for execution on a sequence of inputs 
    to drive the iteration towards convergence.
    """
    @nn.compact
    def __call__(self, x: jnp.ndarray, x0: jnp.ndarray, dense: nn.Module) -> jnp.ndarray:
        """
        Performs one iteration of the inverse fixed-point calculation.

        Args:
            x (jnp.ndarray): The current estimate of the inverse solution ($x_k$).
            x0 (jnp.ndarray): The target value, which is the output of the direct block ($y$).
            dense (nn.Module): The non-linear transformation $f$ (the ResNet block's inner network).

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: The updated estimate $x_{k+1}$ (as $x$) and a duplicate 
                                            of the updated estimate (as $x$), which is required 
                                            by `jax.lax.scan` for passing intermediate results.
        """
        # Fixed-point iteration: $x_{k+1} = x_0 - f(x_k)$
        x = x0 - dense(x)
        return x, x

# ... (imports)

class InvertibleResNet(nn.Module):
    """
    Implements a sequence of Invertible Residual Network (ResNet) blocks, 
    often used as a building block in Normalizing Flows (e.g., Residual Flows).

    The direct (forward) pass is an additive residual map: $y = x + f(x)$.
    The inverse pass is computed numerically via fixed-point iteration 
    (Picard iteration) because the inverse function, $x = y - f(x)$, is implicit.

    Attributes:
        archi (List[List[int]]): A list where each inner list defines the 
                                 architecture (layer sizes) of a single 
                                 NormalizedMultiLayerPerceptron (MLP) block.
        no_inv_iters (Optional[int]): The number of fixed-point iterations to 
                                      perform for the inverse calculation. Defaults to 100.
        lip (Optional[float]): The Lipschitz constant upper bound for the MLP 
                               blocks $f(x)$ (to ensure convergence of the inverse 
                               via Banach's fixed-point theorem). Defaults to 0.9.
        activation (Optional[activationType]): The activation function type used in the MLPs. 
                                               Defaults to `activationType.lipswish`.
        svd (Optional[svdType]): The SVD type used for spectral normalization of the MLPs. 
                                 Defaults to `svdType.direct_indiv`.
    """

    archi: List[List[int]]
    no_inv_iters: Optional[int] = 100
    lip: Optional[float] = .9
    activation: Optional[activationType] = activationType.lipswish
    svd: Optional[svdType] = svdType.direct_indiv

    def setup(self):
        """
        Initializes the list of NormalizedMultiLayerPerceptron blocks (the NNs).
        """
        self.NNs = [NormalizedMultiLayerPerceptron(arch, activation=self.activation, svd=self.svd, lip=self.lip) 
                    for arch in self.archi]
        
    @nn.compact
    def __call__(self, x: jnp.ndarray, mode='direct') -> jnp.ndarray:
        """
        Applies the sequence of ResNet blocks in the direct or inverse mode.

        Args:
            x (jnp.ndarray): The input data.
            mode (evaluationMode, optional): The evaluation mode. Must be 
                                            'direct' or 'inverse'. Defaults to 'direct'.

        Returns:
            jnp.ndarray: The output of the composite transformation.
        """
        if mode == evaluationMode.direct:
            # Direct pass: $y = x + f(x)$
            for block in self.NNs:
                x = block(x) + x
            return x    
            
        elif mode == evaluationMode.inverse:
            # Inverse pass: Numerical calculation of $x = y - f(x)$
            # The blocks are inverted in reverse order.
            for block in reversed(self.NNs):
                x0 = x # $y$ becomes the fixed target for this block's inversion
                
                # JAX scan is used to perform the fixed-point iteration across 'no_inv_iters' steps.
                # The 'x' input to scan is the initial guess ($x_0$), which is $y$.
                units = nn.scan(Inverse, variable_broadcast="params",
                            split_rngs={"params": True}, in_axes=0)
                
                # The scan operation iteratively computes $x_{k+1} = x_0 - f(x_k)$.
                # The sequence of $x_0$ values (the fixed target $y$) is broadcasted.
                # The final converged value is $x$.
                x, _ = units()(x, jnp.array([x0]*self.no_inv_iters), block)
                
        return x


class IResNet(nn.Module):

    a: Union[List[float], NDArray[np.float64]]
    b: Union[List[float], NDArray[np.float64]]
    intervals: NDArray[np.float64]
    xmin: Union[List[float], NDArray[np.float64]]
    xmax: Union[List[float], NDArray[np.float64]]
    features: List[int]

    ## Significant optional quantities
    activations: Optional[List[Callable]] = (
        ActivationFunction.LIPSWISH,
        ActivationFunction.LIPSWISH,
    )
    lipschitz_constant: Optional[float] = 0.9
    opt_a: Optional[bool] = True
    opt_b: Optional[bool] = True
    no_resnet_blocks: Optional[int] = 1
    no_inv_iters: Optional[int] = 30

    ## Specific variables that usually dont require of change
    _wrapper: List[Callable] = Tanh
    fix_point_method: Optional[fixPoint] = fixPoint.REGULAR
    svd_method: Optional[svdType] = svdType.svd_multiple    

    def setup(self):
        self.linear_input = Linear(
            a = 0.995 / (self.xmax - self.xmin) * 2,
            b = -0.995 * (self.xmin + self.xmax) / (self.xmax - self.xmin),
            opt_a=False,
            opt_b=False,
        )
        self.linear_output = LinearOnInterval(
            a=self.a,
            b=self.b,
            intervals=self.intervals,
            opt_a=self.opt_a,
            opt_b=self.opt_b,
        )
        self.linear_ = [
            Linear(
                a=jnp.ones_like(jnp.asarray(self.a)),
                b=jnp.zeros_like(jnp.asarray(self.b)),
                opt_a=True,
                opt_b=True,
            )
            for _ in range(self.no_resnet_blocks)
        ]
        self.wrapper = self._wrapper()

        self.resnet = [
            InvertibleResNetBlock(
                features=self.features,
                activations=self.activations,
                lipschitz_constant=self.lipschitz_constant,
                svd_method=self.svd_method,
                fix_point_method=self.fix_point_method,
                no_inv_iters=self.no_inv_iters,
            )
            for _ in range(self.no_resnet_blocks)
        ]

    @nn.compact
    def __call__(self, x, inverse: bool = False, train: bool = False, return_inter: bool=False):
        if inverse:
            x = self.linear_input(x, mode="direct")#inverse=True)
            x = self.wrapper(x, inverse=True)
            
            for block, linear in zip(reversed(self.resnet), reversed(self.linear_)):
                x = block(x, inverse=True)
                x = linear(x, mode="direct")#inverse=True)
             
            #x = self.wrapper(x, inverse=False)
            
            x = self.linear_output(x, inverse=True)
            
        else:
            x = self.linear_output(x, inverse=False)

            #x = self.wrapper(x, inverse=True)

            for block, linear in zip(self.resnet, self.linear_):
                x = linear(x, mode="inverse") # inverse=False)
                x = block(x, inverse=False)
                
            y = x
            x = self.wrapper(x, inverse=False)
            x = self.linear_input(x, mode="inverse")#inverse=False)
        
        if return_inter:
            return y
        else:
            return x
        

class InvertibleResNetBlock(nn.Module):
    """Single iResNet block MLP(x) + x"""

    features: List[int]
    activations: Optional[Union[List[activationType], activationType]] = (
        activationType.lipswish,
    )
    
    fix_point_method: Optional[fixPoint] = fixPoint.REGULAR
    svd_method: Optional[svdType] = svdType.svd_multiple
    no_inv_iters: Optional[int] = 30
    lipschitz_constant: Optional[float] = 0.9

    def setup(self):
        try:
            self._activations = [
                self.activations[i] if i < len(self.activations) else lambda x: x
                for i in range(len(self.features))
            ]
        except TypeError:
            self._activations = [self.activations for _ in range(len(self.features))]

        if self.svd_method == svdType.svd_multiple:
            self.dense_block = _InvertibleDenseBlockSVD_mult(
                features=self.features,
                activations=self._activations,
                lipschitz_constant=self.lipschitz_constant,
            )
        elif self.svd_method == svdType.direct:
            self.dense_block = _InvertibleDenseBlockSVD(
                features=self.features,
                activations=self._activations,
                lipschitz_constant=self.lipschitz_constant,
            )

        elif self.svd_method == svdType.fourier:
            self.dense_block = _InvertibleDenseBlockFourier(
                features=self.features,
                activations=self._activations,
                lipschitz_constant=self.lipschitz_constant,
            )
        else:
            raise NameError(
                f"The choice of svd_method {self.svd_method} is not an implemented method"
            )
        
        
        if self.fix_point_method == fixPoint.REGULAR:
            def __inverse(self, x):
                units = nn.scan(
                    _Inverse,
                    variable_broadcast="params",
                    variable_carry="batch_stats",
                    split_rngs={"params": True},
                    in_axes=0,
                )
                x_, _ = units()([x,x], jnp.arange(self.no_inv_iters), self.dense_block)
                x, x0 = x_
                return x

        elif self.fix_point_method == fixPoint.HALPERN:
            def __inverse(self, x):
                units = nn.scan(
                    _Inverse_Halpern,
                    variable_broadcast="params",
                    variable_carry="batch_stats",
                    split_rngs={"params": True},
                    in_axes=0,
                )
                x_, _ = units()([x,x], jnp.arange(self.no_inv_iters), self.dense_block)
                x, x0 = x_
                return x

        elif self.fix_point_method == fixPoint.PICARD:
            def __inverse(self, x):
                units = nn.scan(
                    _Inverse_Picardi,
                    variable_broadcast="params",
                    variable_carry="batch_stats",
                    split_rngs={"params": True},
                    in_axes=0,
                )
                x_, _ = units()([x,x], jnp.arange(self.no_inv_iters), self.dense_block)
                x, x0 = x_
                return x
            
        self._inverse  = lambda x: __inverse(self, x)
            
        
    def _direct(self, x):
        return self.dense_block(x) + x
    

    @nn.compact
    def __call__(self, x, inverse: bool = False):
        if inverse:
            return self._inverse(x)
        return self._direct(x)