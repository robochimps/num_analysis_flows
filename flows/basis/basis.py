from functools import partial
from typing import Callable, List, Optional, Tuple

import jax
import numpy as np
from chex import ArrayDevice, dataclass
from jax import config
from jax import numpy as jnp
from numpy.typing import NDArray

config.update("jax_enable_x64", True)


@dataclass
class ProductBasis:

    batch_size_coo: int
    no_batches_coo: int
    batch_ind_coo: ArrayDevice
    coords: ArrayDevice
    weights: ArrayDevice
    multi_ind_coo: ArrayDevice
    padding_size_coo: int
    batch_size_qua: int
    no_batches_qua: int
    batch_ind_qua: ArrayDevice
    quanta: ArrayDevice
    multi_ind_qua: ArrayDevice
    padding_size_qua: int
    list_psi: ArrayDevice
    list_dpsi: ArrayDevice

    @property
    def nbas(self):
        return self.no_batches_qua * self.batch_size_qua - self.padding_size_qua

    @property
    def min_coo(self):
        return jnp.min(self.coo, axis=0)

    @property
    def max_coo(self):
        return jnp.max(self.coo, axis=0)

    @property
    def coo(self):
        if self.padding_size_coo > 0:
            return jnp.concatenate(self.coords, axis=0)[: -self.padding_size_coo]
        else:
            return jnp.concatenate(self.coords, axis=0)

    @property
    def weight(self):
        if self.padding_size_coo > 0:
            return jnp.concatenate(self.weights, axis=0)[: -self.padding_size_coo]
        else:
            return jnp.concatenate(self.weights, axis=0)

    def batch_coo(self, ibatch_coo: int):
        return self.coords[ibatch_coo]

    def batch_weight(self, ibatch_coo: int):
        return self.weights[ibatch_coo]

    def batch_qua(self, ibatch_qua: int):
        return self.quanta[ibatch_qua]

    def batch_psi(self, ibatch_coo: int, ibatch_qua: int):
        return self._product_psi(
            self.multi_ind_coo[ibatch_coo], self.multi_ind_qua[ibatch_qua]
        )

    def batch_dpsi(self, ibatch_coo: int, ibatch_qua: int):
        return self._product_dpsi(
            self.multi_ind_coo[ibatch_coo], self.multi_ind_qua[ibatch_qua]
        )

    def batch_dens(
        self,
        ibatch_coo: int,
        bra_vec: Optional[NDArray[np.float64]] = None,
        ket_vec: Optional[NDArray[np.float64]] = None,
        only_rho: bool = False,
    ):
        psi = jax.lax.scan(
            lambda _, ibatch_qua: (
                0,
                self.batch_psi(ibatch_coo=ibatch_coo, ibatch_qua=ibatch_qua),
            ),
            0,
            self.batch_ind_qua,
        )[1]
        psi = jnp.reshape(jnp.transpose(psi, (1, 0, 2)), (psi.shape[1], -1))

        if bra_vec is not None:
            nbas = len(bra_vec)
            psi1 = jnp.einsum("gi,ij...->gj...", psi[:, :nbas], bra_vec)
            # the number of basis functions `nbas` can be smaller than
            #   the number of `psi` functions, because the latter are padded
            #   with zero functions for equal-size batching.
        else:
            psi1 = psi[:, :, None]

        if ket_vec is not None:
            nbas = len(ket_vec)
            psi2 = jnp.einsum("gi,ij...->gj...", psi[:, :nbas], ket_vec)
        else:
            psi2 = psi[:, :, None]

        psi1_shape = psi1.shape
        psi2_shape = psi2.shape
        psi1 = psi1.reshape(psi1_shape[:2] + (-1,))
        psi2 = psi2.reshape(psi2_shape[:2] + (-1,))
        rho = jnp.einsum("gix,giy->gxy", psi1, psi2)
        rho = rho.reshape((-1,) + psi1_shape[2:] + psi2_shape[2:])
        # 'x' and 'y' here correspond to the (optional) third dimension
        #   in the `bra_vec` and `ket_vec`, respectively, which are used
        #   for auxiliary quantum numbers, such as rotational quanta

        if not only_rho:
            dpsi = jax.lax.scan(
                lambda _, ibatch_qua: (
                    0,
                    self.batch_dpsi(ibatch_coo=ibatch_coo, ibatch_qua=ibatch_qua),
                ),
                0,
                self.batch_ind_qua,
            )[1]
            dpsi = jnp.reshape(
                jnp.transpose(dpsi, (1, 0, 2, 3)), (dpsi.shape[1], -1, dpsi.shape[-1])
            )

            if bra_vec is not None:
                nbas = len(bra_vec)
                dpsi1 = jnp.einsum("gik,ij...->gjk...", dpsi[:, :nbas, :], bra_vec)
            else:
                dpsi1 = dpsi[:, :, :, None]

            if ket_vec is not None:
                nbas = len(ket_vec)
                dpsi2 = jnp.einsum("gik,ij...->gjk...", dpsi[:, :nbas, :], ket_vec)
            else:
                dpsi2 = dpsi[:, :, :, None]

            dpsi1_shape = dpsi1.shape
            dpsi2_shape = dpsi2.shape
            dpsi1 = dpsi1.reshape(dpsi1_shape[:3] + (-1,))
            dpsi2 = dpsi2.reshape(dpsi2_shape[:3] + (-1,))

            rho1_a = jnp.einsum("gikx,giy->gkxy", dpsi1, psi2)
            rho1_b = jnp.einsum("gix,giky->gxky", psi1, dpsi2)
            rho2 = jnp.einsum("gikx,gily->gkxly", dpsi1, dpsi2)
            # 'x' and 'y' are (optional) auxiliary quanta,
            #   corresponding to the third dimension of the `bra_vec`
            #   and `ket_vec`, respectively

            rho1_a = rho1_a.reshape((-1,) + dpsi1_shape[2:] + psi2_shape[2:])
            rho1_b = rho1_b.reshape((-1,) + psi1_shape[2:] + dpsi2_shape[2:])
            rho2 = rho2.reshape((-1,) + dpsi1_shape[2:] + dpsi2_shape[2:])

            # when auxiliary dimensions 'x' and 'y' are not empty
            if rho1_b.ndim > 2:
                rho1_b = rho1_b.swapaxes(1, 2)
            if rho2.ndim > 3:
                rho2 = rho2.swapaxes(2, 3)

            return rho, rho1_a, rho1_b, rho2

        return rho

    def _product_psi(
        self,
        multi_ind_coo: NDArray[np.int_],
        multi_ind_qua: NDArray[np.int_],
    ):
        psi = jnp.prod(
            jnp.asarray(
                [
                    self.list_psi[icoo][
                        jnp.ix_(multi_ind_coo[icoo, :], multi_ind_qua[icoo, :])
                    ]
                    for icoo in range(len(self.list_psi))
                ]
            ),
            axis=0,
        )
        return psi

    def _product_dpsi(
        self,
        multi_ind_coo: NDArray[np.int_],
        multi_ind_qua: NDArray[np.int_],
    ):
        dpsi = jnp.prod(
            jnp.asarray(
                [
                    [
                        (
                            self.list_dpsi[icoo][
                                jnp.ix_(multi_ind_coo[icoo, :], multi_ind_qua[icoo, :])
                            ]
                            if icoo == jcoo
                            else self.list_psi[jcoo][
                                jnp.ix_(multi_ind_coo[jcoo, :], multi_ind_qua[jcoo, :])
                            ]
                        )
                        for icoo in range(len(self.list_psi))
                    ]
                    for jcoo in range(len(self.list_psi))
                ]
            ),
            axis=0,
        )
        return jnp.transpose(dpsi, (1, 2, 0))


@dataclass
class ProductBasisDynamicGrid:

    list_quanta: List[List[int]]
    batch_size_qua: int
    no_batches_qua: int
    batch_ind_qua: ArrayDevice
    quanta: ArrayDevice
    multi_ind_qua: ArrayDevice
    padding_size_qua: int
    list_psi: List[Callable[[ArrayDevice, ArrayDevice], ArrayDevice]]
    list_dpsi: List[Callable[[ArrayDevice, ArrayDevice], ArrayDevice]]

    def batch_psi(self, coords, ibatch_qua: int):
        # precompute 1D basis functions on grid `coords[no_points, no_coords]`
        list_psi = [
            f(x, n)
            for f, x, n in zip(self.list_psi, coords.T, self.list_quanta)
        ]
        multi_ind_coo = jnp.arange(len(coords))
        multi_ind_qua = self.multi_ind_qua[ibatch_qua]

        psi = jnp.prod(
            jnp.asarray(
                [
                    psi_icoo[jnp.ix_(multi_ind_coo, ind_qua)]
                    for ind_qua, psi_icoo in zip(multi_ind_qua, list_psi)
                ]
            ),
            axis=0,
        )

        return psi

    def batch_dpsi(
        self,
        coords,
        ibatch_qua: NDArray[np.int_],
    ):

        multi_ind_coo = jnp.arange(len(coords))
        multi_ind_qua = self.multi_ind_qua[ibatch_qua]
        
        list_psi = [
            f(x, n)
            for f, x, n in zip(self.list_psi, coords.T, self.list_quanta)
            ]
        
        list_dpsi = [
            f(x, n)
            for f, x, n in zip(self.list_dpsi, coords.T, self.list_quanta)
        ]


        dpsi = jnp.prod(
            jnp.asarray(
                [
                    [
                        (
                            list_dpsi[icoo][
                                jnp.ix_(multi_ind_coo, multi_ind_qua[icoo])
                            ]
                            if icoo == jcoo
                            else list_psi[jcoo][
                                jnp.ix_(multi_ind_coo, multi_ind_qua[jcoo])
                            ]
                        )
                        for icoo in range(len(self.list_psi))
                    ]
                    for jcoo in range(len(self.list_psi))
                ]
            ),
            axis=0,
        )
        return jnp.transpose(dpsi, (1, 2, 0))

class Basis:
    def __init__(
        self,
        list_psi: List[Callable[[NDArray[np.float64], int], NDArray[np.float64]]],
        select_quanta: Callable[[List[int]], bool],
        select_coords: Callable[[NDArray[np.float64], NDArray[np.float64]], bool],
        complex: bool = False,
    ):
        self.list_psi = [jax.jit(jax.vmap(bas, in_axes=(0, None))) for bas in list_psi]
        if complex:
            self.list_dpsi = [
            jax.jit(jax.vmap(jax.jacrev(bas, argnums=0, holomorphic=True), in_axes=(0, None)))
            for bas in list_psi
        ]
        else:
            self.list_dpsi = [
            jax.jit(jax.vmap(jax.jacrev(bas, argnums=0), in_axes=(0, None)))
            for bas in list_psi
            ]
        self.select_quanta = select_quanta
        self.select_coords = select_coords
        self.ncoo = len(self.list_psi)

    def product_basis(
        self,
        list_coords: List[NDArray[np.float64]],
        list_weights: List[NDArray[np.float64]],
        list_quanta: List[List[int]],
        batch_size_coo: int = None,
        batch_size_qua: int = None,
        padding_coo: float = 1.0,
        padding_weight: float = 0,
        padding_qua: int = 0,
        coords_and_weights: Tuple[NDArray[np.float64], NDArray[np.float64]] = None,
    ) -> ProductBasis:

        if batch_size_coo is not None and batch_size_coo <= 0:
            batch_size_coo = None

        if batch_size_qua is not None and batch_size_qua <= 0:
            batch_size_qua = None

        if coords_and_weights is None:
            # generate product grid

            coords = []
            weights = []
            multi_ind_coo = []

            for coo, wght, m_ind in _generate_product_coords(
                list_coords, list_weights, self.select_coords, batch_size_coo
            ):
                # here, the size of batches may be different
                #   because self.select_coords function may prune different
                #   number of points from different batches
                coords.append(coo)
                weights.append(wght)
                multi_ind_coo.append(m_ind)

            coords = np.concatenate(coords, axis=0)
            weights = np.concatenate(weights, axis=0)
            multi_ind_coo = np.concatenate(multi_ind_coo, axis=-1)

        else:
            coords, weights = coords_and_weights
            list_coords, multi_ind_coo = unravel_grid(coords)

        # split prunned grid into equal-size batches

        coords_batches = []
        weights_batches = []
        multi_ind_coo_batches = []

        tot_size_coo = len(coords)
        if batch_size_coo is None or batch_size_coo >= tot_size_coo:
            batch_size_coo = tot_size_coo
        no_batches_coo = (tot_size_coo + batch_size_coo - 1) // batch_size_coo

        for ibatch_coo in range(no_batches_coo):
            start_ind = ibatch_coo * batch_size_coo
            end_ind = jnp.minimum(start_ind + batch_size_coo, tot_size_coo)

            coords_batch = coords[start_ind:end_ind]
            weights_batch = weights[start_ind:end_ind]
            multi_ind_coo_batch = multi_ind_coo[:, start_ind:end_ind]

            # if batch is smaller than batch_size_coo, pad the indices ...
            if no_batches_coo > 1:
                padding_size_coo = batch_size_coo - multi_ind_coo_batch.shape[-1]
            else:
                padding_size_coo = 0
            if padding_size_coo > 0:
                # index of the padding point is the last+1 point in the array for each coordinate
                #   this last+1 point will be added to the arrays of psi and dpsi (see below)
                padding_ind = jnp.array([len(coo) for coo in list_coords])
                multi_ind_coo_batch = jnp.hstack(
                    (
                        multi_ind_coo_batch,
                        jnp.tile(padding_ind, (padding_size_coo, 1)).T,
                    )
                )
                coords_batch = jnp.pad(
                    coords_batch,
                    ((0, padding_size_coo), (0, 0)),
                    constant_values=padding_coo,
                )
                weights_batch = jnp.pad(
                    weights_batch, (0, padding_size_coo), constant_values=padding_weight
                )

            coords_batches.append(coords_batch)
            weights_batches.append(weights_batch)
            multi_ind_coo_batches.append(multi_ind_coo_batch)

        # generate product quanta

        quanta = []
        multi_ind_qua = []

        for qua, m_ind in _generate_product_quanta(
            list_quanta, self.select_quanta, batch_size_qua
        ):
            # here, the size of batches may be different
            #   because self.select_quanta function may prune different
            #   number of qunta combinations from different batches
            quanta.append(qua)
            multi_ind_qua.append(m_ind)

        quanta = np.concatenate(quanta, axis=0)
        multi_ind_qua = np.concatenate(multi_ind_qua, axis=-1)

        # split prunned quanta into equal-size batches

        quanta_batches = []
        multi_ind_qua_batches = []

        tot_size_qua = len(quanta)
        if batch_size_qua is None or batch_size_qua >= tot_size_qua:
            batch_size_qua = tot_size_qua
        no_batches_qua = (tot_size_qua + batch_size_qua - 1) // batch_size_qua

        for ibatch_qua in range(no_batches_qua):
            start_ind = ibatch_qua * batch_size_qua
            end_ind = jnp.minimum(start_ind + batch_size_qua, tot_size_qua)

            quanta_batch = quanta[start_ind:end_ind]
            multi_ind_qua_batch = multi_ind_qua[:, start_ind:end_ind]

            # if batch is smaller than batch_size_qua, pad the indices with 0
            if no_batches_qua > 1:
                padding_size_qua = batch_size_qua - multi_ind_qua_batch.shape[-1]
            else:
                padding_size_qua = 0
            if padding_size_qua > 0:
                # index of the padding point is the last+1 point in the array for each set of quanta
                #   this last+1 point will be added to the arrays of psi and dpsi (see below)
                padding_ind = jnp.array([len(qua) for qua in list_quanta])
                multi_ind_qua_batch = jnp.hstack(
                    (
                        multi_ind_qua_batch,
                        jnp.tile(padding_ind, (padding_size_qua, 1)).T,
                    )
                )
                quanta_batch = jnp.pad(
                    quanta_batch,
                    ((0, padding_size_qua), (0, 0)),
                    constant_values=padding_qua,
                )

            quanta_batches.append(quanta_batch)
            multi_ind_qua_batches.append(multi_ind_qua_batch)

        # precompute 1D basis functions

        list_psi = [
            f(x, jnp.arange(0, jnp.max(n) + 1))[:, n]
            for f, x, n in zip(self.list_psi, list_coords, list_quanta)
        ]

        list_dpsi = [
            f(x, jnp.arange(0, jnp.max(n) + 1))[:, n]
            for f, x, n in zip(self.list_dpsi, list_coords, list_quanta)
        ]

        # append last+1 point along the coordinate and quantum number dimensions
        #   these last points are added for correct padding (see above)
        list_psi = [
            jnp.pad(psi, ((0, 1), (0, 1)), mode="constant", constant_values=0)
            for psi in list_psi
        ]
        list_dpsi = [
            jnp.pad(dpsi, ((0, 1), (0, 1)), mode="constant", constant_values=0)
            for dpsi in list_dpsi
        ]

        return ProductBasis(
            batch_size_coo=batch_size_coo,
            no_batches_coo=len(coords_batches),
            batch_ind_coo=jnp.arange(len(coords_batches)),
            coords=jnp.array(coords_batches),
            weights=jnp.array(weights_batches),
            multi_ind_coo=jnp.array(multi_ind_coo_batches),
            padding_size_coo=padding_size_coo,
            batch_size_qua=batch_size_qua,
            no_batches_qua=len(quanta_batches),
            batch_ind_qua=jnp.arange(len(quanta_batches)),
            quanta=jnp.array(quanta_batches),
            multi_ind_qua=jnp.array(multi_ind_qua_batches),
            padding_size_qua=padding_size_qua,
            list_psi=list_psi,
            list_dpsi=list_dpsi,
        )
    
    def product_basis_dynamic_grid(
        self,
        list_quanta: List[List[int]],
        batch_size_qua: int = None,
        padding_qua: int = 0,
    ):

        if batch_size_qua is not None and batch_size_qua <= 0:
            batch_size_qua = None

        # generate product quanta

        quanta = []
        multi_ind_qua = []

        for qua, m_ind in _generate_product_quanta(
            list_quanta, self.select_quanta, batch_size_qua
        ):
            # here, the size of batches may be different
            #   because self.select_quanta function may prune different
            #   number of qunta combinations from different batches
            quanta.append(qua)
            multi_ind_qua.append(m_ind)

        quanta = np.concatenate(quanta, axis=0)
        multi_ind_qua = np.concatenate(multi_ind_qua, axis=-1)

        # split prunned quanta into equal-size batches

        quanta_batches = []
        multi_ind_qua_batches = []

        tot_size_qua = len(quanta)
        if batch_size_qua is None or batch_size_qua >= tot_size_qua:
            batch_size_qua = tot_size_qua
        no_batches_qua = (tot_size_qua + batch_size_qua - 1) // batch_size_qua

        for ibatch_qua in range(no_batches_qua):
            start_ind = ibatch_qua * batch_size_qua
            end_ind = jnp.minimum(start_ind + batch_size_qua, tot_size_qua)

            quanta_batch = quanta[start_ind:end_ind]
            multi_ind_qua_batch = multi_ind_qua[:, start_ind:end_ind]

            # if batch is smaller than batch_size_qua, pad the indices with 0
            if no_batches_qua > 1:
                padding_size_qua = batch_size_qua - multi_ind_qua_batch.shape[-1]
            else:
                padding_size_qua = 0
            if padding_size_qua > 0:
                # index of the padding point is the last+1 point in the array for each set of quanta
                #   this last+1 point will be added to the arrays of psi and dpsi (see below)
                padding_ind = jnp.array([len(qua) for qua in list_quanta])
                multi_ind_qua_batch = jnp.hstack(
                    (
                        multi_ind_qua_batch,
                        jnp.tile(padding_ind, (padding_size_qua, 1)).T,
                    )
                )
                quanta_batch = jnp.pad(
                    quanta_batch,
                    ((0, padding_size_qua), (0, 0)),
                    constant_values=padding_qua,
                )

            quanta_batches.append(quanta_batch)
            multi_ind_qua_batches.append(multi_ind_qua_batch)

        return ProductBasisDynamicGrid(
            list_quanta=list_quanta,
            batch_size_qua=batch_size_qua,
            no_batches_qua=len(quanta_batches),
            batch_ind_qua=jnp.arange(len(quanta_batches)),
            quanta=jnp.array(quanta_batches),
            multi_ind_qua=jnp.array(multi_ind_qua_batches),
            padding_size_qua=padding_size_qua,
            list_psi=self.list_psi,
            list_dpsi=self.list_dpsi,
        )


def _generate_product_coords(
    list_coords: List[NDArray[np.float64]],
    list_weights: List[NDArray[np.float64]],
    select_coords: Callable[[NDArray[np.float64], NDArray[np.float64]], bool],
    batch_size: int = None,
):
    lengths = [len(coo) for coo in list_coords]
    tot_size = jnp.prod(jnp.asarray(lengths))
    if batch_size is None:
        batch_size = tot_size
    no_batches = (tot_size + batch_size - 1) // batch_size

    for ibatch in range(no_batches):
        start_ind = ibatch * batch_size
        end_ind = jnp.minimum(start_ind + batch_size, tot_size)
        batch_ind = jnp.arange(start_ind, end_ind)
        multi_ind = jnp.array(jnp.unravel_index(batch_ind, lengths))

        coords = jnp.array(
            [list_coords[icoo][multi_ind[icoo, :]] for icoo in range(len(list_coords))]
        ).T
        weights = jnp.prod(
            jnp.array(
                [
                    list_weights[icoo][multi_ind[icoo, :]]
                    for icoo in range(len(list_weights))
                ]
            ),
            axis=0,
        )
        ind = jnp.where(
            jnp.asarray([select_coords(x, w) for x, w in zip(coords, weights)])
        )
        yield coords[ind], weights[ind], multi_ind[:, ind[0]]


def _generate_product_quanta(
    list_quanta: List[NDArray[np.int_]],
    select_quanta: Callable[[List[int]], bool],
    batch_size: int = None,
):
    lengths = [len(coo) for coo in list_quanta]
    tot_size = jnp.prod(jnp.asarray(lengths))
    if batch_size is None:
        batch_size = tot_size
    no_batches = (tot_size + batch_size - 1) // batch_size

    for ibatch in range(no_batches):
        start_ind = ibatch * batch_size
        end_ind = jnp.minimum(start_ind + batch_size, tot_size)
        batch_ind = jnp.arange(start_ind, end_ind)
        multi_ind = jnp.array(jnp.unravel_index(batch_ind, lengths))
        quanta = jnp.array(
            [list_quanta[icoo][multi_ind[icoo, :]] for icoo in range(len(list_quanta))]
        ).T
        ind = jnp.where(jnp.asarray([select_quanta(q) for q in quanta]))
        yield quanta[ind], multi_ind[:, ind[0]]


def unravel_grid(grid):
    ncoo = grid.shape[-1]
    x_and_ind = [np.unique(grid[:, icoo], return_inverse=True) for icoo in range(ncoo)]
    list_coords = [elem[0] for elem in x_and_ind]
    multi_ind = np.array([elem[1] for elem in x_and_ind])
    return list_coords, multi_ind