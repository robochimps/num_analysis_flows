import os
from functools import partial
from scipy import constants
import joblib
import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import config
from numpy.polynomial.hermite import hermgauss
import os 
from flows.basis.basis import Basis
import matplotlib.pyplot as plt 
from numpy.polynomial.hermite import hermgauss
from flows.Bases import Hermite 
from flows.types import *
from flows.bases import *
from flows.models.linear import Linear
import math 
import pickle 
from flows.models.iresnet import IResNet, ActivationFunction
from scipy.special import genlaguerre, gamma, gammaln 

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Computer Modern Roman'
fontsize_label = 18
fontsize_title = fontsize_label + 1
fontsize_legend = fontsize_label - 4
# size of the ticks 
plt.rcParams['xtick.labelsize'] = fontsize_label - 5
plt.rcParams['ytick.labelsize'] = fontsize_label - 5


mS, mO, mN, mC, mH = [31.97207070,15.994914619257, 14.003074004251,12.0,1.00782503223]
m1 = mH
m2 = mO
a_m,D_m,x0_m = 2.1440, 42301, 0.0
no_states = 23
#molecule and potential and kinetic energy operators

def potential(x):
    return (D_m * (1.0 - jnp.exp(-a_m * (x - x0_m)))**2)[:,0]#.reshape(-1,1)

#potential = jax.jit(jax.vmap(lambda x: pot(x,a_m,D_m,x0_m)))

G_to_invcm = (
    constants.value("Planck constant")
    * constants.value("Avogadro constant")
    * 1e16
    / (4.0 * np.pi**2 * constants.value("speed of light in vacuum"))
    * 1e5
)

def abs_det_jac_x(model, params, x):
    def det(params):
        def det(x):
            return jnp.abs(jnp.linalg.det(jax.jacrev(model.apply, 1)(params, x)))
        return jax.vmap(det, in_axes=0)(x)
    return jax.jit(det)(params)

# G-matrix for the Morse potential
def gmat(x):
    npoints = x.shape[0]
    gvib = (m1+m2)/(m1*m2) * jax.numpy.ones_like(x)[:,:,None] * G_to_invcm
    grot = jnp.zeros((npoints,))
    gcor = jnp.zeros((npoints, 1))
    return gvib, grot, gcor


def lambda_compute():
    D_si = D_m * 100 * constants.h * constants.c  # cm⁻¹ to J
    a_si = a_m * 1e10                         # Å⁻¹ to m⁻¹
    hbar = constants.hbar                        # J·s
    amu = constants.u                      # atomic mass unit in kg
    mu = m1 * m2 / (m1 + m2) * amu  
    # Compute lambda
    lambda_ = jnp.sqrt(2 * mu * D_si) / (a_si * hbar)
    return lambda_

def sol(x, n):
    # Convert inputs to SI units
    x_m = x * 1e-10                                # Å to m
    a_si = a_m * 1e10                              # 1/Å to 1/m
    D_si = D_m * constants.h * constants.c * 100   # cm⁻¹ to J
    mu = m1 * m2 / (m1 + m2) * constants.atomic_mass  # reduced mass in kg
    hbar = constants.hbar                          # J·s

    nu_e = a_si / (jnp.pi * constants.c * 100) * jnp.sqrt(D_si/(2*mu)) #Harmonic wavenumber in cm-1
    X_e = a_si**2 * hbar / (4*jnp.pi * mu * constants.c * 100 * nu_e) #Anharmonicity constant (unitless)

    beta_n = 1/X_e - 2 * n - 1 #beta_n is unitless
    y = 1/X_e * jnp.exp(-a_m * x) #a_m in A^-1 and x in A.
    
    log_N_n = 0.5 * (jnp.log(a_m) +jnp.log(beta_n) +gammaln(n + 1) - gammaln(beta_n + n + 1))
    log_exp = -0.5 * y                
    log_pow = 0.5 * beta_n * jnp.log(y)  
    L_n = genlaguerre(n, beta_n)(y)  
    log_f = log_N_n + log_exp + log_pow
    f = jnp.exp(log_f) * L_n
    """
    D_si = D_m * 100 * constants.h * constants.c  # cm⁻¹ to J
    a_si = a_m * 1e10                         # Å⁻¹ to m⁻¹
    hbar = constants.hbar                        # J·s
    amu = constants.u                      # atomic mass unit in kg
    mu = m1 * m2 / (m1 + m2) * amu  
    # Compute lambda
    lambda_ = jnp.sqrt(2 * mu * D_si) / (a_si * hbar)
    z = 2*lambda_*jnp.exp(-x)
    alpha = 2*lambda_-2*n-1

    coeff = 1#jnp.sqrt(scipy.special.factorial(n)*a_si*alpha/scipy.special.gamma(2*lambda_-n))
    f = coeff * z**(lambda_-n-0.5) * jnp.exp(-0.5*z) * (genlaguerre(n, alpha)(z))
    #norm = (f[:,0]**2*w).sum()
    """
    return f

def decay(x):
    D_si = D_m * 100 * constants.h * constants.c  # cm⁻¹ to J
    a_si = a_m * 1e10                         # Å⁻¹ to m⁻¹
    hbar = constants.hbar                        # J·s
    amu = constants.u                      # atomic mass unit in kg
    mu = m1 * m2 / (m1 + m2) * amu  
    # Compute lambda
    lambda_ = jnp.sqrt(2 * mu * D_si) / (a_si * hbar)
    y = (lambda_ - .5)*(jnp.log(2*lambda_) - x) - lambda_ * jnp.exp(-x)
    jac = 1/jnp.sqrt(abs_det_jac_x(model, params, x))
    return (y[:,0]*jac)**2
 
if __name__ == "__main__":
    x, w = hermgauss(100)
    x = x.reshape(-1,1)
    xmin = np.min(x, axis=0)
    xmax = np.max(x, axis=0)    
    
    nmax = 22
    n_basis = [nmax for _ in range(x.shape[1])]
    w_basis = [1 for _ in range(x.shape[1])]
    basis_r = Hermite.init_basis(n_basis, w_basis, nmax, orthotype=orthoType.ortho) # 

    psi_o = partial(Hermite.batch_basis_values, basis_r)  

    dpsi_o = partial(Hermite.batch_dbasis_values, basis_r)
    
    nblocks = 5
    pmax = 23
    
    x_leg = jnp.linspace(-10.5,10.5,82)
    # training and testing basis sets
    
    # Linear model 

    # flow model
    a = jnp.array([1.43630802])
    b = jnp.array([-1.4624287])

    model_lin = Linear(a=a, b=b)

    model = IResNet(
        a=a,
        b=b,
        intervals=jnp.array([[-jnp.inf, jnp.inf]]),
        xmin=xmin,
        xmax=xmax,
        features=[8, 8, 1],
        activations=[
                ActivationFunction.LIPSWISH,
                ActivationFunction.LIPSWISH,
                ActivationFunction.LIPSWISH, #
                ],
        no_resnet_blocks=nblocks)
    
    print("Model:\n", model)

    
    with open("simulations_data/Morse_NF_Hermite.pkl", "rb") as f:
        params = pickle.load(f)['params']

    with open("simulations_data/Morse_scaled_Hermite.pkl", "rb") as f: 
        params_scaled = pickle.load(f)['params']
     
    model_lin_r = lambda params, x: model_lin.apply(params, x[:,0].reshape(-1,1), mode="direct")
 
    model_lin_x = lambda params, x: (
        model_lin.apply(params, x[:, 0].reshape(-1, 1), mode="inverse")
        if x.ndim == 2 else
        model_lin.apply(params, jnp.array(x[0]).reshape(1, -1), mode="inverse")[0, :]
        )
    
    model_r = lambda params, x: model.apply(params, x[:,0].reshape(-1,1), inverse=True)

    model_x = lambda params, x: (
        model.apply(params, x[:, 0].reshape(-1, 1), inverse=False)
        if x.ndim == 2 else
        model.apply(params, jnp.array(x[0]).reshape(1, -1), inverse=False)[0, :]
        )
        

     #x_plot = np.linspace(-8,8,1000)
    a_lin =  0.08222986 
    b_lin = -7.44875051e-09
    fig, axs = plt.subplots(1, 1, figsize=(4,4)) 
    
    r_plot = model_r(params, x.reshape(-1,1))
    r_lin = a_lin * x + b_lin 
    r_opt_lin = model_lin_r(params_scaled, x.reshape(-1,1))
    pot_ev = potential(r_lin.reshape(-1,1))
    pot_f_ev = potential(r_plot.reshape(-1,1))
    pot_lin_ev = potential(r_opt_lin.reshape(-1,1))
    """
    axs[1].plot(x, pot_ev, linestyle="--", color="#0173B2", label="id")
    axs[1].plot(x, pot_lin_ev, color='#D55E00', label="Linear")
    axs[1].plot(x, pot_f_ev, color="#029E73", label="iResNet")
    axs[1].grid(True, which='major', linestyle='--', alpha=0.6)
    axs[1].set_xlabel(r"$x$", fontsize=fontsize_label)
    axs[1].set_ylabel(r"$V \circ g^{-1} \ (x), \ \mathrm{cm}^{-1}$", fontsize=fontsize_label)
    
    axs[1].legend(fontsize=fontsize_legend)
    axs[1].set_ylim(0, 50000)
    axs[1].set_xlim(-8,8)
    """
    r = model_r(params, x)
    rmin = np.min(r, axis=0)
    rmax = np.max(r, axis=0)
    # loss and test functions
    no_states = 10

    def _grad_log_abs_det_jac_x(model, params, x_batch, **kwargs):
        def det(x):
            return jnp.log(
            jnp.abs(jnp.linalg.det(jax.jacrev(model, argnums=1)(params, x, **kwargs)))
        )

        return jax.vmap(jax.grad(det), in_axes=0)(x_batch)
    

    def _jac_x(model, params, x_batch, **kwargs):
        def jac(x):
            return jax.jacrev(model, argnums=1)(params, x, **kwargs)

        return jax.vmap(jac, in_axes=0)(x_batch)

    
    def hamiltonian(model_fs, params, x, psi1, dpsi1, psi2, dpsi2, w):
        model_r_f, model_x_f, model = model_fs
        r = model_r_f(params, x)
        p = potential(r)
        ovlp = jnp.zeros(p.shape)
        gvib, _, _ = gmat(r)
        dlog_det = _grad_log_abs_det_jac_x(model_x_f, params, r)
        
        df = _jac_x(model_x_f, params, r)
        gvib1 = jnp.einsum("gka,gab...,glb->gkl...", df, gvib, df)
        gvib2 = 0.5 * jnp.einsum("gka,gab...,gb->gk...", df, gvib, dlog_det)
        gvib3 = 0.5 * jnp.einsum("ga,gab...,gkb->gk...", dlog_det, gvib, df)
        gvib4 = 0.25 * jnp.einsum("ga,gab...,gb->g...", dlog_det, gvib, dlog_det)

        print("Shapes in hamiltonian: ", dpsi1.shape, gvib1.shape, dpsi2.shape, w.shape)
        #exit()
        keo_vib = (
                jnp.einsum("gik,gkl...,gjl,g->ij...", dpsi1, gvib1, dpsi2, w)
                + jnp.einsum("gik,gk...,gj,g->ij...", dpsi1, gvib2, psi2, w)
                + jnp.einsum("gi,gk...,gjk,g->ij...", psi1, gvib3, dpsi2, w)
                + jnp.einsum("gi,g...,gj,g->ij...", psi1, gvib4, psi2, w)
            )
        poten = jnp.einsum("gi,gj,g...,g->ij...", psi1, psi2, p, w)
        overlap = jnp.einsum("gi,gj,g...,g->ij...", psi1, psi2, ovlp, w)
        ham = 0.5 * (keo_vib ) + poten
        return ham

    def eigensolve(models, par, psi1, dpsi1, psi2, dpsi2, no_states):
        h = hamiltonian(models, par, x, psi1, dpsi1, psi2, dpsi2, w)
        e, v = jax.numpy.linalg.eigh(h)
        return e[:no_states], v
    eigensolve_jit = eigensolve#partial(jax.jit, static_argnums=(0,6))(eigensolve)

    _psi = psi_o(x)
    _dpsi_ = dpsi_o(x)
    
    # iResNet case 
    e, v = eigensolve_jit((model_r, model_x, model), params, _psi, _dpsi_, _psi, _dpsi_, no_states)
    
    e_lin, v_lin = eigensolve_jit((model_lin_r, model_lin_x, model_lin), params_scaled, _psi, _dpsi_, _psi, _dpsi_, no_states)


    r_grid = jnp.linspace(rmin[0], rmax[0], 10000)
    #r_grid = jnp.linspace(-8., 8., 10000)
    ww = np.zeros_like(r_grid)
    dx = np.diff(r_grid)
    ww[1:-1] = (dx[:-1] + dx[1:])/2
    ww[0] = dx[0]/2
    ww[-1] = dx[-1]/2
    r_grid = r_grid[:,None]
    
    x = model_x(params, r_grid)
    x_2 = model_x(params, r_grid)
    #r = model_r(params, x)
    
    psis_approx = jnp.einsum('jk,km->jm', _psi, v) # alternatively ljk,mk->ljm
    n = 0
    omega = 10
    psi_true = sol(r_grid[:,0], n)
    norm = jnp.sqrt(jnp.sum(ww * psi_true**2)) 
    psi_true /= norm
    
    #linear case 
    
    
    
    x_lin = model_lin_x(params_scaled, r_grid)
    x_lin_2 = model_lin_x(params_scaled, r_grid)
    #r = model_r(params, x)
    
    psis_approx = jnp.einsum('jk,km->jm', _psi, v_lin) # alternatively ljk,mk->ljm
    n = 0
    omega = 10
    

    den_true = psi_true**2
    
    psi_true_c = sol(x_2[:,0], n)
    psi_true_c /= norm
    grad = jnp.sqrt(abs_det_jac_x(model, params, r_grid,))# return_inter=True))
    psi_true_transformed = psi_true_c*grad
    den_true_transformed = psi_true_transformed**2
    
    psi_true_lin = sol(x_lin_2[:,0], n)
    psi_true_lin /= norm
    grad_lin = jnp.sqrt(abs_det_jac_x(model_lin, params_scaled, r_grid,))# return_inter=True))
    psi_true_lin_transformed = psi_true_lin*grad_lin
    den_true_lin_transformed = psi_true_lin_transformed**2
    
    gaussian = jnp.exp(-r_grid[:,0]**2)
    gaussian /= jnp.sqrt((gaussian**2*ww).sum())
    g_den = gaussian**2 

    integ_os, integ_ts, integ_gs = [], [], []
    integ_lin = []
    
    omegas = jnp.arange(4)
    for omega in omegas:
        integ_o = (jnp.abs(r_grid[:,0])**omega*den_true*ww).sum() 
        integ_s = (jnp.abs(r_grid[:,0])**omega*den_true_transformed*ww).sum()
        integ_g = (jnp.abs(r_grid[:,0])**omega*g_den*ww).sum()
        integ_lin_val = (jnp.abs(r_grid[:,0])**omega*den_true_lin_transformed*ww).sum()
        
        integ_os.append(integ_o)
        integ_ts.append(integ_s)
        integ_gs.append(integ_g)
        integ_lin.append(integ_lin_val)

    axs.plot(omegas, integ_os, marker="^", color="#0173B2", label="id")
    axs.plot(omegas, integ_ts, marker="o", color="#029E73", label="iResNet")
    axs.plot(omegas, integ_gs, marker="x", color="black", label="Gaussian")
    axs.plot(omegas, integ_lin, marker="s", color="#D55E00", label="Linear")
    axs.grid(True, which='major', linestyle='--', alpha=0.6)
    axs.set_xticks(omegas)
    axs.set_xlabel(r"$s$", fontsize=fontsize_label)
    axs.set_ylabel(r"$I^2(s)$", fontsize=fontsize_label)
    axs.set_yscale('log')
    axs.legend(fontsize=fontsize_legend)
    plt.subplots_adjust(wspace=0.3) 
    axs.set_xlabel(r"$s$", fontsize=fontsize_label)
    axs.set_xlabel(r"$x$", fontsize=fontsize_label)

    plt.savefig("moments.pdf", dpi=300, bbox_inches='tight')
    sys.exit()
    psi_approx = psis_approx[:,n]
    psi_approx /= jnp.sqrt((psi_approx**2*w1).sum())
    # Construct analytical solution

    plt.plot(x, psi_approx, label="Approx")
    plt.plot(x, psi_true, label="True")
    plt.legend()
    plt.show()
    sys.exit()


