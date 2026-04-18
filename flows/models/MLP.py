from typing import List, Optional, Union
from flows.types import svdType
import jax
from jax import config
from jax import numpy as jnp
from flax import linen as nn
from numpy.typing import NDArray
import numpy as np
import flows 
from ..types import activationType
from ..utils import lipswish, sinh, cosh, _erf, scaled_erf

config.update("jax_enable_x64", True)

def spectral_norm(kernel: jnp.ndarray) -> float:
    """
    Computes the spectral norm (largest singular value) of the given kernel.

    Args:
        kernel (jnp.ndarray): The weight matrix/kernel for which to compute the spectral norm.

    Returns:
        float: The spectral norm (largest singular value) of the kernel.
    """
    return jnp.max(jnp.linalg.svd(kernel, compute_uv=False))


def clip_kernel_svd_multiple(params, lipschitz_constant): 
    """
    Enforce Lipschitz constant via individual singular values (SVD multiple block).

    Parameters
    ----------
    params : pytree
        Model parameters.
    lipschitz_constant : float
        Maximum allowed singular value.

    Returns
    -------
    pytree
        Parameters with singular values clipped.
    """

    def clip_fn(path, leaf):
        key = getattr(path[-1], 'key', None)
        if isinstance(key, str) and 'w_' in key:
            L, S, R = jnp.linalg.svd(leaf, full_matrices=False)
            S_clipped = jnp.minimum(S, lipschitz_constant)
            return (L * S_clipped) @ R
        return leaf
    return jax.tree_util.tree_map_with_path(clip_fn, params)


def _svd_fourier(kernel: jnp.ndarray, input_shape: tuple, lip: float) -> jnp.ndarray:
    """
    Normalizes the kernel (weight matrix) in the Fourier domain by scaling 
    the largest singular value of the transformed kernel to be $\leq$ lip. 
    This is typically used for convolutional kernels.

    Args:
        kernel (jnp.ndarray): The weight matrix/kernel to be normalized.
        input_shape (tuple): The shape to which the kernel is FFT-transformed.
        lip (float): The target Lipschitz constant (spectral norm upper bound).

    Returns:
        jnp.ndarray: The spectrally normalized kernel.
    """
    transforms = jnp.fft.fft2(kernel, input_shape, axes=[0, 1])
    sv = jnp.max(jnp.linalg.svd(transforms, compute_uv=False))
    # If sv > lip, scale the kernel: $\text{kernel} \cdot (\text{lip} / \text{sv})$
    return jax.lax.cond(sv >= lip, lambda a, b: lip * a / b,
                                   lambda a, b: a, kernel, sv)

def _svd_direct(kernel: jnp.ndarray, input_shape: tuple, lip: float) -> jnp.ndarray:
    """
    Normalizes the kernel by directly calculating its spectral norm (largest 
    singular value, $\sigma_{\text{max}}$) and scaling the entire kernel if $\sigma_{\text{max}} > \text{lip}$.

    Args:
        kernel (jnp.ndarray): The weight matrix/kernel to be normalized.
        input_shape (tuple): Placeholder argument for API consistency; ignored in this method.
        lip (float): The target Lipschitz constant (spectral norm upper bound).

    Returns:
        jnp.ndarray: The spectrally normalized kernel.
    """
    sv = jnp.max(jnp.linalg.svd(kernel, compute_uv=False))
    # If sv > lip, scale the kernel: $\text{kernel} \cdot (\text{lip} / \text{sv})$
    return jax.lax.cond(sv >= lip, lambda a, b: lip * a / b,
                                   lambda a, b: a, kernel, sv)


def _svd_direct_indiv(kernel: jnp.ndarray, input_shape: tuple, lip: float) -> jnp.ndarray:
    """
    Normalizes the kernel by truncating its singular values individually. 
    Any singular value $s_i$ greater than $\text{lip}$ is replaced by $\text{lip}$.
    This preserves more information in the weight matrix compared to scaling the entire kernel.

    Args:
        kernel (jnp.ndarray): The weight matrix/kernel to be normalized.
        input_shape (tuple): Placeholder argument for API consistency; ignored in this method.
        lip (float): The target Lipschitz constant (singular value upper bound).

    Returns:
        jnp.ndarray: The spectrally normalized kernel with truncated singular values.
    """
    L,S,R = jnp.linalg.svd(kernel, full_matrices=False)
    
    # Truncate singular values: $s'_i = \min(s_i, \text{lip})$
    def update_S(x):
        return jax.lax.cond(x >= lip,  lambda _: lip,
                            lambda _: x,  None,)
    S_updated = jax.vmap(update_S)(S)
    
    # Reconstruct the kernel: $K' = L \Sigma' R^T$
    return jnp.dot(L * S_updated, R)


class NormalizedMultiLayerPerceptron(nn.Module):
    """
    A class representing a multi-layer perceptron (MLP) where the weight
    matrices are normalized to have a spectral norm $\sigma_{\text{max}} \leq \text{lip}$.
    
    This normalization is crucial for enforcing the **Lipschitz continuity** of the 
    network, which is required for guaranteeing invertibility or convergence 
    in models like Residual Networks or Normalizing Flows.

    Attributes:
        n_hidden_units (List[int]): A list of integers representing the number of 
                                 hidden units in each layer. The list length defines the number of layers.
        activation (activationType): The activation function type to be used (e.g., relu, sigmoid, lipswish).
        svd (svdType): The type of singular value decomposition (SVD) used for kernel 
                       normalization (e.g., direct, fourier, direct_indiv).
        lip (float): The target Lipschitz constant (spectral norm upper bound) 
                     for the weight matrices.
    """

    n_hidden_units: List[int]
    activation: activationType
    svd: svdType
    lip: float

    def setup(self):
        """
        Set up the MLP by initializing the SVD function based on the specified SVD type.
        """
        if self.svd == svdType.fourier:
            self.svd_func = _svd_fourier
        elif self.svd == svdType.direct:
            self.svd_func = _svd_direct
        elif self.svd == svdType.direct_indiv:
            self.svd_func = _svd_direct_indiv
        else:
            raise ValueError("Invalid SVD type specified.")

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of the MLP.

        Args:
            x (jnp.ndarray): The input tensor.

        Returns:
            jnp.ndarray: The output tensor after passing through the MLP.
        """
        size_ = x.shape[-1]
        for i, size in enumerate(self.n_hidden_units):
            # Define kernel and bias parameters
            kernel = self.param(f'w_{i}', jax.nn.initializers.glorot_uniform(),
                                (size_, size))
            bias = self.param(f'b_{i}', jax.nn.initializers.zeros,(size,))
            
            # Normalize Kernel to enforce Lipschitz constraint
            # comment to shift the normalization to the update step, which is more efficient for training
            #kernel = self.svd_func(kernel, jnp.shape(kernel), self.lip)
            
            # Linear transformation
            x = jnp.dot(x, kernel) + bias            
            
            # Apply activation function, but skip for the last layer
            if i < len(self.n_hidden_units) - 1:
                
                if self.activation == activationType.relu:    
                    x = nn.relu(x)
            
                elif self.activation == activationType.sigmoid:
                    x = nn.sigmoid(x)
            
                elif self.activation == activationType.lipswish:
                    x = lipswish(x)
                
                elif self.activation == activationType.sinh:
                    x = sinh(x)

                elif self.activation == activationType.cosh:
                    x = cosh(x)

                elif self.activation == activationType.erf:
                    x = _erf(x)

                elif self.activation == activationType.scaled_erf:
                    x = scaled_erf(x)
                else:
                    raise ValueError("Invalid activation function")

            size_ = size
        
        return x