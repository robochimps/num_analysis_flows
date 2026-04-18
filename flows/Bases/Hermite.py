import numpy as np
import itertools
from collections import namedtuple
from typing import List, Optional
from numpy.typing import NDArray
from jax import numpy as jnp
#from jax.config import config
#config.update("jax_enable_x64", True)
import jax
import math 
from flows.bases import orthoType

def init_basis(n: List[int], w: List[float], nmax: Optional[float] = None,
               nind = None, q: Optional[List[List[int]]] = None,
               vec: Optional[List[NDArray[np.float64]]] = None,
               div_by_r_pow: int = 0, orthotype=orthoType.ortho):

    assert (len(n) == len(w)), f"len(n) != len(w): {len(n)} =! {len(w)}"
    
    if nind is None:
        nind = [np.arange(nn+1) for nn in n]
    assert (len(nind) == len(w)), f"len(nind) != len(w): {len(nind)} =! {len(w)}"

    if q is None:
        if nmax is None:
            quanta = [elem for elem in itertools.product(*nind)]
        else:
            #quanta = [elem for elem in itertools.product(*nind)
            #            if np.sum(np.multiply(elem, w)) <= nmax]
            direct_product = np.array(np.meshgrid(*nind)).reshape(len(n), -1).T 
            mask = np.sum(np.multiply(np.array(w),direct_product), axis=1) <= nmax
            quanta = direct_product[mask]
    else:
        quanta = [elem for elem in q]
    quanta = np.array(quanta)

    sqsqpi = np.sqrt(np.sqrt(np.pi))
    if orthotype == orthoType.ortho:
        if vec == None:
            norm = [np.diag([1.0 / np.sqrt(2.0**n * math.factorial(n)) / sqsqpi
                         for n in range(np.max(elem)+1)])
                for elem in quanta.T]
        else:
            assert (len(n) == len(vec)), f"len(n) != len(vec): {len(n)} =! {len(vec)}"
            norm = [np.array([1.0 / np.sqrt(2.0**n * math.factorial(n)) / sqsqpi * v[n, :]
                         for n in range(elem+1)])
                for elem, v in zip(n, vec)]
    
    elif orthotype == orthoType.proj:
        # Basis for projection
        norm = [np.diag([1.0 / 2.0**n * math.factorial(n) / (sqsqpi**2)
                         for n in range(np.max(elem)+1)])
                for elem in quanta.T]

    elif orthotype == orthoType.n_ortho:
        norm = [np.diag([1.0 
                         for _ in range(np.max(elem)+1)])
                for elem in quanta.T]

    bas = {'quanta': quanta, 'nbas': len(quanta), 'norm': norm, 'div_by_r_pow': div_by_r_pow}
    bas = namedtuple('basis', bas.keys())(*bas.values())
    return bas

@jax.jit
def basis_values(bas, x):
    """Product basis of Hermite functions,
    taking out the product of exponents
    """
    c = [bas.norm[icoo] for icoo in range(len(x))]
    herm1d = [_hermval(xx, cc) for xx, cc in zip(x, c)]
    herm = jnp.prod(jnp.asarray([h[i] for i, h in zip(bas.quanta.T, herm1d)]),
                    axis=0)
    r = jnp.linalg.norm(x, axis=-1)
    herm = herm #/ jnp.power(r, bas.div_by_r_pow)
    return herm

batch_basis_values = jax.jit(jax.vmap(basis_values, in_axes=(None, 0)))

@jax.jit
def basis_jac(bas, x):
    """Jacobian of the prodcut of Hermite functions,
    taking out the product of exponents
    """
    jac = _basis_jac(bas, x)
    val = batch_basis_values(bas, x)
    return jac - val[:, :, None] * x[:, None, :]

@jax.jit
def dbasis_values(bas, x): 
    """
    Jacobian of the product of Hermite functions, 
    computed using jac.fwd on 1Ds and constructing 
    the direct product manually, 
    taking out the product of exponents
    """
    c = [bas.norm[icoo] for icoo in range(x.shape[-1])]
    herm1d = [_hermval(xx, cc) for xx, cc in zip(x, c)]
    dherm1d = [_dhermval(xx,cc) - xx*hermval
               for xx, cc, hermval in zip(x,c,herm1d)] 
    jac = []
    for icoo, dh in enumerate(dherm1d):
        flist = [dh if jcoo == icoo else f for jcoo, f in enumerate(herm1d)]
        prod = jnp.prod(jnp.array([f[i] for f, i in zip(flist, bas.quanta.T)]), axis=0)
        jac.append(prod)
    return jnp.array(jac).T

batch_dbasis_values = jax.jit(jax.vmap(dbasis_values, in_axes=(None, 0)))

@jax.jit
def basis_hess(bas, x):
    """Hessian of the prodcut of Hermite functions,
    taking out the product of exponents
    """
    hess = _basis_hess(bas, x)
    jac = _basis_jac(bas, x)
    val = batch_basis_values(bas, x)
    ncoo = x.shape[-1]
    return hess - jac[:, :, :, None] * x[:, None, None, :] \
        - jac[:, :, None, :] * x[:, None, :, None] \
        + val[:, :, None, None] * x[:, None, :, None] * x[:, None, None, :] \
        - val[:, :, None, None] * jnp.eye(ncoo)[None, None, :, :]


@jax.jit
def _basis_jac(bas, x_batch):
    def jac(x_batch):
        def jac(x):
            return jax.jacfwd(basis_values, 1)(bas, x)
        return jax.vmap(jac, in_axes=0)(x_batch)
    return jac(x_batch)

@jax.jit
def _dhermval(x, c):
    return jax.jacfwd(_hermval, 0)(x,c)

@jax.jit
def _basis_hess(bas, x_batch):
    def hess(x_batch):
        def hess(x):
            return jax.jacfwd(jax.jacfwd(basis_values, 1), 1)(bas, x) # forward(forward) works fastest
            # return jax.hessian(basis_values, 1)(bas, x)
        return jax.vmap(hess, in_axes=0)(x_batch)
    return hess(x_batch)


@jax.jit
def dens_values(bas, ibas, x):
    val = basis_values(bas, x) * jnp.prod(jnp.exp(-0.5 * x**2), axis=-1)
    d = val**2
    return d[ibas]


batch_dens_values = jax.jit(jax.vmap(dens_values, in_axes=(None, None, 0)))


@jax.jit
def _hermval(x, c):
    def iter(carry, cc):
        c0, c1, nd = carry
        tmp = c0
        nd = nd - 1
        c0 = cc - c1*(2*(nd - 1))
        c1 = tmp + c1*x2
        return (c0, c1, nd), 0
    c = c.reshape(c.shape + (1,)*x.ndim)
    x = jnp.asarray(x)
    x2 = x*2
    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        # use scan instead of the loop below
        carry, _ = jax.lax.scan(iter, (c0, c1, nd), np.flip(c, axis=0)[2:])
        c0, c1, nd = carry
    return c0 + c1*x2
