import os
os.environ['JAX_PLATFORMS'] = 'cpu'
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
from flows.basis.hermite_custom_jvp import hermite
from flows.hamiltonian import hamiltonian, hamiltonian_trace, eigenvalues
from flows.models.invertible_block import ActivationFunction
from flows.models.models import IResNet2, IResNet, Linear, SingularValues
config.update("jax_enable_x64", True)

mS, mO, mN, mC, mH = [31.97207070,15.994914619257, 14.003074004251,12.0,1.00782503223]
m1 = mH
m2 = mO
a_m, D_m, x0_m = 2.1440, 42301, 0.0
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


def gmat(x):
    npoints = x.shape[0]
    gvib = (m1+m2)/(m1*m2) * jax.numpy.ones_like(x)[:,:,None] * G_to_invcm
    grot = jnp.zeros((npoints,))
    gcor = jnp.zeros((npoints, 1))
    return gvib, grot, gcor

if __name__ == "__main__":
    restart = 0 #0 = no restart, 1 = restart from pmax16, 2 = restart from latest iteration
    nblocks = 5#int(sys.argv[1])
    pmax = int(sys.argv[1])
    ckpt_dir = f"Morse_res_{pmax}"
    
    # training and testing basis sets
    basis = Basis(
        [hermite],
        lambda q: np.sum(q * np.array([1])) <= pmax,
        lambda x, w: True, complex=False,
    )

    list_quanta = [np.arange(pmax+1)]

    batch_size_coo = 10582
    batch_size_qua = 1000000
    no_train_sets = 1

    no_points_per_set = [100+n for n in range(0, no_train_sets)]
    no_points_per_set += [100]  # add test set
    basis_set = []

    for iset, n1 in enumerate(no_points_per_set):
        x1, w1 = hermgauss(n1)
        w1 /= np.exp(-(x1**2))

        list_coords = [x1]
        list_weights = [w1]

        bas = basis.product_basis(
            list_coords,
            list_weights,
            list_quanta,
            batch_size_coo=batch_size_coo,
            batch_size_qua=batch_size_qua,
        )
        basis_set.append(bas)

        print(
            f"basis set no. {iset}   ",
            "no. functions:",
            sum(len(elem) for elem in bas.quanta),
            f"(padding: {bas.padding_size_qua}, no. batches: {len(bas.quanta)})   ",
            "no. points:",
            sum(len(elem) for elem in bas.coords),
            f"(padding: {bas.padding_size_coo}, no. batches: {len(bas.coords)})",
        )

    # flow model

    xmin = np.min([bas.min_coo for bas in basis_set], axis=0)
    xmax = np.max([bas.max_coo for bas in basis_set], axis=0)
    print(
        "Min and max values of quadrature coords accross all basis sets:\n", xmin, xmax
    )
    
    a = jnp.array([1.43630802])
    b = jnp.array([-1.4624287])

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

    x = basis_set[0].coo
    
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    if restart == 0:
        print(x.shape, "**")
        params = model.init(jax.random.PRNGKey(0), x[:,0].reshape(-1,1))
        epoch_start = 0
    
    elif restart == 1:
        print(f"restart with parameters stored in folder transf_paper_results/H2S")
        params = joblib.load("Morse_res_23/"+'params.json') #Load from other folder
        #params = joblib.load("h2s_se_iresnet2_nblocks_10_pmax_12/"+'params.json') 
        epoch_start = 0
    
    else:
        print(f"restart from the latest-epoch parameters stored in folder '{ckpt_dir}'")
        params = joblib.load(f"{ckpt_dir}/"+'params.json') #Load from own folder
        with open(f"{ckpt_dir}/"+'loss') as f:
            for line in f:
                pass
            epoch_start = int(line.split('loss')[0])

    print('Starting at epoch: ', epoch_start)
    model_r = lambda params, x: model.apply(params, x[:,0].reshape(-1,1), inverse=True)
 
    model_x = lambda params, x: (
        model.apply(params, x[:, 0].reshape(-1, 1), inverse=False)
        if x.ndim == 2 else
        model.apply(params, jnp.array(x[0]).reshape(1, -1), inverse=False)[0, :]
            )
    r = model_r(params, x)#jnp.linspace(-1.395, 1.558, 1000).reshape(-1,1)
    
    #model_r = lambda params, x: model.apply(params, x, inverse=True)
    #model_x = lambda params, x: model.apply(params, x, inverse=False)
    rmin = np.min(
        [np.min(model_r(params, bas.coo), axis=0) for bas in basis_set], axis=0
    )
    rmax = np.max(
        [np.max(model_r(params, bas.coo), axis=0) for bas in basis_set], axis=0
    )
    print("Min and max values of physical coords accross all basis sets:\n", rmin, rmax)
    # loss and test functions
    no_states = 10

    def eigensolve(par, bas, nbas, no_states):
        h, s = hamiltonian(par, model_x, model_r, bas, nbas, gmat, potential, pseudo_func=None)
        e, w1, w2 = eigenvalues(h)
        return e[:no_states], w1, w2

    eigensolve_jit = partial(jax.jit, static_argnums=(2, 3))(eigensolve)

    def loss_grad_fn(par, bas, nbas, no_states):
        h, s = hamiltonian(par, model_x, model_r, bas, nbas, gmat, potential, pseudo_func=None)
        e, w1, w2 = eigenvalues(h)
        return jax.value_and_grad(hamiltonian_trace)(
            par,
            model_x,
            model_r,
            bas,
            nbas,
            gmat,
            potential,
            pseudo_func=None,
            eigenvec=w1[:, :no_states],
            eigenvec_h=w2[:, :no_states],
        )

    print("First few eigenvalues on a test set using initial params:")
    bas = basis_set[-1]
    e,_,_ = eigensolve_jit(params, bas, bas.nbas, no_states)
    print('Loss',jnp.sum(e))
    print(e[0], e[:10] - e[0])
    
    # optimisation
    optx = optax.adam(learning_rate=0.001)
    opt_state = optx.init(params)

    @partial(jax.jit, static_argnums=(2,))
    def update_params(params, opt_state, no_states):
        for bas in basis_set[:-1]:
            loss_val, grad = loss_grad_fn(params, bas, bas.nbas, no_states)
            updates, opt_state = optx.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
        return loss_val, params, opt_state

    print(f"input 'pmax' = {pmax}")
    out_file = open(f"{ckpt_dir}/energies", "a")
    loss_file = open(f"{ckpt_dir}/loss", "a")
    for i in range(epoch_start,1001): 
        loss_val, params, opt_state = update_params(params, opt_state, no_states)
        print(i, loss_val)

        if i % 30 == 0:
            bas = basis_set[-1]
            r = model_r(params, bas.coo)
            rmin = np.min(r, axis=0) 
            rmax = np.max(r, axis=0) 
            e,w1,w2 = eigensolve_jit(params, bas, bas.nbas, no_states)
            
            loss_val = np.sum(e[:no_states])
            print("Test loss:", loss_val)
            print("Min and max values of coords:\n", rmin, rmax)
            print("First few eigenvalues on a test set:\n", e[0], e[:10] - e[0])
            x_inv_r = model_x(params, r)
            error_inv = np.max(np.abs(bas.coo-x_inv_r))
            print('Error inverse:',error_inv)
            out_file.write(
                f"{i:6d} " + " ".join(f"{elem:18.12f}" for elem in e[:no_states]) + "\n"
            )
            out_file.flush()
            os.fsync(out_file)

            print_loss = "  ".join(
                f"{i}" + " loss %20.12f"%loss_val \
                    + " err inv %1.2e"%error_inv \
                    + " rmin " + " ".join("%12.6f"%elem for elem in r1) \
                    + " rmax " + " ".join("%12.6f"%elem for elem in r2)
                for (r1, r2) in zip([rmin], [rmax])
            )
            loss_file.write(print_loss + "\n")
            loss_file.flush()
            os.fsync(loss_file)
            joblib.dump(params, f"{ckpt_dir}/"+'params.json')
            print(f"store updated parameters in folder '{ckpt_dir}'")
            np.save("w1.npy", w1)
            np.save("w2.npy", w2)
    out_file.close()
