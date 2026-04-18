import jax.numpy as jnp 
import flax.linen as nn 
import jax 
from jax.scipy.special import erf

def _erf(x):
    return (1/1.1) * 0.5 * jnp.sqrt(jnp.pi) * erf(x)

def scaled_erf(x):
    return (1/1.245) * x * (1 + erf(x)) / 2

def sinh(x: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the hyperbolic sine of the input array element-wise.

    The hyperbolic sine function is defined as:
    $\sinh(x) = \frac{e^x - e^{-x}}{2}$.

    Args:
        x (jnp.ndarray): The input array.

    Returns:
        jnp.ndarray: An array containing the hyperbolic sine of each element in the input.
    """
    return (jnp.exp(x) - jnp.exp(-x)) / 2.0

def cosh(x: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the hyperbolic sine of the input array element-wise.

    The hyperbolic sine function is defined as:
    $\sinh(x) = \frac{e^x - e^{-x}}{2}$.

    Args:
        x (jnp.ndarray): The input array.

    Returns:
        jnp.ndarray: An array containing the hyperbolic sine of each element in the input.
    """
    return (jnp.exp(x) + jnp.exp(-x)) / 2.0

def lipswish(x: jnp.ndarray) -> jnp.ndarray:
    """
    Implements a scaled version of the Swish activation function, designed 
    to be 1-Lipschitz (or $\alpha$-Lipschitz with $\alpha \approx 1/1.1$) 
    for use in contractive maps like Residual Flows.

    The function is defined as: $f(x) = \frac{x}{1.1} \cdot \sigma(x)$, 
    where $\sigma(x)$ is the sigmoid function.

    Args:
        x (jnp.ndarray): The input tensor.

    Returns:
        jnp.ndarray: The output tensor after applying the lipswish activation.
    """
    return (x / 1.1) * nn.sigmoid(x)

def abs_det_jac_x(model: nn.Module, params, x_batch: jnp.ndarray, has_aux: bool = False, **kwargs) -> jnp.ndarray:
    """
    Computes the absolute determinant of the Jacobian matrix for the transformation 
    $f(\mathbf{x}) = \text{model.apply}(\mathbf{x})$ across a batch of inputs. 
    This is essential for change-of-variables (e.g., calculating the log-likelihood 
    in Normalizing Flows).

    The calculation is $|\det(\mathbf{J})| = |\det(\nabla_{\mathbf{x}} f(\mathbf{x}))|$.

    Args:
        model (nn.Module): The Flax model to evaluate.
        params: The learned parameters of the model.
        x_batch (jnp.ndarray): A batch of input vectors (B, D).
        has_aux (bool, optional): Passed to `jax.jacrev`. True if `model.apply` returns auxiliary data.
        **kwargs: Additional keyword arguments passed to `model.apply`.

    Returns:
        jnp.ndarray: An array of absolute determinant values, shape (B,).
    """
    def det(params):
            def det_single(x):
                # Calculate Jacobian using reverse-mode autodiff
                jac = jax.jacrev(model.apply, argnums=1, has_aux=has_aux)(params, x, **kwargs)
                jac = jac[0] if has_aux else jac
                return jnp.abs(jnp.linalg.det(jac))
            return jax.vmap(det_single, in_axes=0)(x_batch)
    return jax.jit(det)(params)

def s_abs_det_jac_x(model: nn.Module, params, x_batch: jnp.ndarray, has_aux: bool = False, **kwargs) -> jnp.ndarray:
    """
    Computes the square root of the absolute determinant of the Jacobian 
    of the model's transformation across a batch of inputs. This is often 
    used in density calculations as part of the transformation kernel.

    The calculation is $\sqrt{|\det(\mathbf{J})|}$.

    Args:
        model (nn.Module): The Flax model to evaluate.
        params: The learned parameters of the model.
        x_batch (jnp.ndarray): A batch of input vectors (B, D).
        has_aux (bool, optional): Passed to `jax.jacrev`. True if `model.apply` returns auxiliary data.
        **kwargs: Additional keyword arguments passed to `model.apply`.

    Returns:
        jnp.ndarray: An array of $\sqrt{|\det(\mathbf{J})|}$ values, shape (B,).
    """
    def det(params):
            def det_single(x):
                jac = jax.jacrev(model.apply, argnums=1, has_aux=has_aux)(params, x, **kwargs)
                jac = jac[0] if has_aux else jac
                return jnp.sqrt(jnp.abs(jnp.linalg.det(jac)))
            return jax.vmap(det_single, in_axes=0)(x_batch)
    return jax.jit(det)(params)

def jac_x(model: nn.Module, params, x_batch: jnp.ndarray, **kwargs) -> jnp.ndarray:
    """
    Computes the full Jacobian matrix $\mathbf{J} = \nabla_{\mathbf{x}} f(\mathbf{x})$ 
    for the model's transformation across a batch of inputs.

    Args:
        model (nn.Module): The Flax model to evaluate.
        params: The learned parameters of the model.
        x_batch (jnp.ndarray): A batch of input vectors (B, D).
        **kwargs: Additional keyword arguments passed to `model.apply`.

    Returns:
        jnp.ndarray: An array of Jacobian matrices, shape (B, D, D).
    """
    def jac(params):
        def jac_single(x):
            # argnums=1 specifies derivative w.r.t. the second argument (x)
            jac_result = jax.jacrev(model.apply, argnums=1, has_aux=False)(params, x, **kwargs)
            # Assuming model.apply returns (output) or (output, aux)
            return jac_result[0] if isinstance(jac_result, tuple) else jac_result
        return jax.vmap(jac_single, in_axes=0)(x_batch)
    return jax.jit(jac)(params)


def hess_x(model: nn.Module, params, x_batch: jnp.ndarray, **kwargs) -> jnp.ndarray:
    """
    Computes the Hessian matrix $\mathbf{H}$ of the model's output with 
    respect to the input $\mathbf{x}$ across a batch of inputs. 
    
    For a vector-valued function $f: \mathbb{R}^D \to \mathbb{R}^D$, 
    the result is a tensor of shape (B, D, D, D).

    Args:
        model (nn.Module): The Flax model to evaluate.
        params: The learned parameters of the model.
        x_batch (jnp.ndarray): A batch of input vectors (B, D).
        **kwargs: Additional keyword arguments passed to `model.apply`.

    Returns:
        jnp.ndarray: An array of Hessian tensors, shape (B, D, D, D).
    """
    def hess(params):
        def hess_single(x):
            # argnums=1 specifies derivative w.r.t. the second argument (x)
            hess_result = jax.hessian(model.apply, argnums=1, has_aux=False)(params, x, **kwargs)
            # Assuming model.apply returns (output) or (output, aux)
            return hess_result[0] if isinstance(hess_result, tuple) else hess_result
        return jax.vmap(hess_single, in_axes=0)(x_batch)
    return jax.jit(hess)(params)

def grad_abs_det_jac_x(model: nn.Module, params, x_batch: jnp.ndarray, **kwargs) -> jnp.ndarray:
    """
    Computes the gradient of the absolute determinant of the Jacobian 
    ($\nabla_{\mathbf{x}} |\det(\mathbf{J})|$) across a batch of inputs.

    Args:
        model (nn.Module): The Flax model to evaluate.
        params: The learned parameters of the model.
        x_batch (jnp.ndarray): A batch of input vectors (B, D).
        **kwargs: Additional keyword arguments passed to `model.apply`.

    Returns:
        jnp.ndarray: An array of gradient vectors, shape (B, D).
    """
    def grad(params):
        def det(x):
            # Compute $|\det(\mathbf{J})|$
            jac = jax.jacrev(model.apply, argnums=1, has_aux=False)(params, x, **kwargs)
            jac = jac[0] if isinstance(jac, tuple) else jac
            return jnp.abs(jnp.linalg.det(jac))
        # jax.grad(det) computes $\nabla_{\mathbf{x}} |\det(\mathbf{J})|$
        return jax.vmap(jax.grad(det), in_axes=0)(x_batch)
    return jax.jit(grad)(params)

def s_grad_abs_det_jac_x(model: nn.Module, params, x_batch: jnp.ndarray, **kwargs) -> jnp.ndarray:
    """
    Computes the gradient of the square root of the absolute determinant of the Jacobian 
    ($\nabla_{\mathbf{x}} \sqrt{|\det(\mathbf{J})|}$) across a batch of inputs.

    Args:
        model (nn.Module): The Flax model to evaluate.
        params: The learned parameters of the model.
        x_batch (jnp.ndarray): A batch of input vectors (B, D).
        **kwargs: Additional keyword arguments passed to `model.apply`.

    Returns:
        jnp.ndarray: An array of gradient vectors, shape (B, D).
    """
    def grad(params):
        def det(x):
            # Compute $\sqrt{|\det(\mathbf{J})|}$
            jac = jax.jacrev(model.apply, argnums=1, has_aux=False)(params, x, **kwargs)
            jac = jac[0] if isinstance(jac, tuple) else jac
            return jnp.sqrt(jnp.abs(jnp.linalg.det(jac)))
        # jax.grad(det) computes $\nabla_{\mathbf{x}} \sqrt{|\det(\mathbf{J})|}$
        return jax.vmap(jax.grad(det), in_axes=0)(x_batch)
    return jax.jit(grad)(params) 
