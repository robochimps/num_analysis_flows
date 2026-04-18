from typing import List, Optional, Union
from jax import config
from jax import numpy as jnp
from flax import linen as nn
from numpy.typing import NDArray
import numpy as np
from ..types import evaluationMode

class Id(nn.Module):
    """
    An Identity neural network module implemented using Flax. 
    
    This module performs the identity function, $f(x) = x$, for any input $x$. 
    It is useful as a placeholder, a bypass, or as the initial or final layer 
    in complex architectures like Normalizing Flows or multi-step models where 
    the transformation should sometimes be null.
    
    This module contains no parameters or internal state.
    """
    @nn.compact
    def __call__(self, x: jnp.ndarray, mode='direct') -> jnp.ndarray:
        """
        Applies the identity function.

        The `mode` argument is accepted for API consistency with other flow layers 
        (e.g., in `CompositeModel`), but it has no effect on the output.

        Args:
            x (jnp.ndarray): The input data.
            mode (evaluationMode, optional): The evaluation mode. Ignored. 
                                            Defaults to 'direct'.

        Returns:
            jnp.ndarray: The input data `x` itself, unchanged.
        """
        return x