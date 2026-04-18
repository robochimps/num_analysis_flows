from typing import List, Optional, Union
from jax import config
from jax import numpy as jnp
from flax import linen as nn
from numpy.typing import NDArray

config.update("jax_enable_x64", True)

class CompositeModel(nn.Module):
    """
    A Flax module that chains together a list of other Flax modules, 
    allowing for sequential composition in a 'direct' mode and 
    reverse composition in an 'inverse' mode. 
    
    This is commonly used in constructing deep architectures like 
    Normalizing Flows, where the forward pass is a sequence of transformations
    and the inverse pass is the reverse sequence of inverse transformations.

    Attributes:
        models (List[nn.Module]): A list of Flax modules to be applied sequentially.
    """

    models: List[nn.Module]

    @nn.compact
    def __call__(self, x: jnp.ndarray, mode='direct') -> jnp.ndarray:
        """
        Applies the sequence of models to the input data 'x'.

        Args:
            x (jnp.ndarray): The input data to the composite model.
            mode (evaluationMode, optional): The evaluation mode. 
                Must be 'direct' (forward pass) or 'inverse' (reverse pass). 
                Defaults to 'direct'.

        Raises:
            ValueError: If an invalid mode is provided.

        Returns:
            jnp.ndarray: The output of the composite transformation.
        """
        # Note: 'evaluationMode' is assumed to be an imported enum.
        # Replacing with string checks for a self-contained example.
        
        if mode == 'direct':
            # Direct Mode: Apply models sequentially (M1 -> M2 -> ... -> Mn)
            for model in self.models:
                x = model(x)
        elif mode == 'inverse':
            # Inverse Mode: Apply models in reverse order, calling their inverse transformation
            # (Mn_inv -> ... -> M2_inv -> M1_inv)
            for model in reversed(self.models):
                # Assumes that each model in self.models implements an inverse mode
                x = model(x, mode='inverse')
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'direct' or 'inverse'.")
        return x