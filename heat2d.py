"""
NOTE: Running this script may take several hours, depending on the computing power of your machine.
To run this script, you need to install the dcgp-python package available via the Conda package manager, see also
    https://github.com/darioizzo/dcgp


We look for exact or approximate closed-form solutions of the heat equation on a ring in polar coordinates (r, phi)
using weighted dCGP expressions.
An expression is deemed a solution of the equation if its fitness falls below a prescribed threshold.
We run a specified number of experiments.
We increase performance by the following means:
 - Run experiments in parallel as a pool of processes
 - Discard an experiment if its fitness doesn't fall fast enough
 - Take a coarse grid for fitness evaluation
 - Take a small kernel set to choose from
 - Take small numbers of dCGP columns and levels back
 - Set the coefficient for thermal diffusivity to 1.

Though taking a small kernel set looks like cheating, solutions are also found with the general kernel set
    +-*/, gaussian, sqrt, log, sin, cos, exp - it just takes a higher number of experiments:
8021			21	[0.0795774715459477*exp(-0.25*r**2/t)/t]
    2.059833516372035e-35

Though setting the coefficient for thermal diffusivity to 1 looks like cheating, too, it is common to do so
in math literature, and doesn't pose any particular issue. Introducing the coefficient as another grid variable
is straightforward. Again - it would just take a higher number of experiments to find a solution that incorporates
that coefficient.

If a solution is not printed in one run, you might want to increase the number of experiments
- or just run the script again.
"""

from typing import Tuple
from multiprocessing import Pool
import numpy as np
from dcgpy.core import expression_weighted_gdual_vdouble as expression
from dcgpy.core import kernel_set_gdual_vdouble as kernel_set
from pyaudi.core import gdual_vdouble as gdual
import pyaudi.core as pyaudi
from sympy import init_printing
from sympy import diff, exp
from sympy.abc import r, phi, t

# We run nexp experiments.
nexp = int(1e4)
# The number of offsprings to consider in each generation.
offsprings = 4
# max number of generations
stop = 100
# A dCGP expression is accepted as a solution if its fitness falls below this threshold.
fitness_lower = 1e-7
# kernels = kernel_set(['sum', 'mul', 'div', 'diff', 'gaussian', 'sqrt', 'log', 'sin', 'cos', 'exp'])()
kernels = kernel_set(['sum', 'mul', 'div', 'diff', 'exp'])()

init_printing()
np.seterr(all='ignore')  # avoids numpy complaining about early on malformed expressions being evaluated

# We construct the grid of points.
nr, nphi, nt = 3, 3, 3
n = nr * nphi * nt
ra = 0.1
rb = 3.
ta = 0.1
tb = 3.
rpoints = np.linspace(ra, rb, nr)
phipoints = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
tpoints = np.linspace(ta, tb, nt)
rval, phival, tval = (i.ravel() for i in np.meshgrid(rpoints, phipoints, tpoints))
gr = gdual(rval, 'r', 2)
gphi = gdual(phival, 'phi', 2)
gt = gdual(tval, 't', 1)
gr0 = gdual(gr.constant_cf)
gphi0 = gdual(gphi.constant_cf)
gt0 = gdual(gt.constant_cf)
grid = [gr, gphi, gt]
gra = gdual([ra] * n)
grb = gdual([rb] * n)
gta = gdual([ta] * n)
gtb = gdual([tb] * n)
# We prescribe grid boundary values.
ura = 1 / (4 * np.pi * gt0) * pyaudi.exp(- gra ** 2 / (4 * gt0))
urb = 1 / (4 * np.pi * gt0) * pyaudi.exp(- grb ** 2 / (4 * gt0))
uta = 1 / (4 * np.pi * gta) * pyaudi.exp(- gr0 ** 2 / (4 * gta))
utb = 1 / (4 * np.pi * gtb) * pyaudi.exp(- gr0 ** 2 / (4 * gtb))
# We discard an experiment if over generations the fitness doesn't fall fast enough.
fitness_upper = 1e2
drop_exp = 2
drop = [fitness_upper / (g + 1) ** drop_exp for g in range(stop)]


def get_diff(dcgp: expression) -> gdual:
    """
    Returns a gdual of an MSE-like error of the dCGP expression in the grid points.
    It incorporates penalty terms for the differential equation violation and for the boundary-conditions violation.
    """
    u = dcgp(grid)[0]
    dudr = gdual(u.get_derivative({'dr': 1}))
    dudr2 = gdual(u.get_derivative({'dr': 2}))
    dudphi2 = gdual(u.get_derivative({'dphi': 2}))
    inv_gr0 = 1 / gr0
    laplace_u = dudr2 + inv_gr0 * (dudr + inv_gr0 * dudphi2)
    dudt = gdual(u.get_derivative({'dt': 1}))
    # Let's not accept trivial terms.
    if str(dudt) == '0' or str(laplace_u) == '0':
        return gdual([1e32])
    diff_eq = dudt - laplace_u
    diff_ra = dcgp([gra, gphi0, gt0])[0] - ura
    diff_rb = dcgp([grb, gphi0, gt0])[0] - urb
    diff_ta = dcgp([gr0, gphi0, gta])[0] - uta
    diff_tb = dcgp([gr0, gphi0, gtb])[0] - utb
    return (diff_eq * diff_eq
            + (diff_ra * diff_ra + diff_rb * diff_rb + diff_ta * diff_ta + diff_tb * diff_tb) / 4
            ) / n


def collapse_vectorized_coefficient(x, N):
    """
    This is used to sum over the component of a vectorized coefficient, accounting for the fact that if its dimension
    is 1, then it could represent [a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a ...] with [a].
    """
    if len(x) == N:
        return sum(x)
    return x[0] * N


def newton(ex: expression) -> None:
    """
    Newton's method for minimizing the error function f w.r.t. the weights of the dCGP expression.
    We take a specified amount of steps, each by choosing randomly 2 or 3 weights.
    """
    n = ex.get_n()
    r = ex.get_rows()
    c = ex.get_cols()
    a = ex.get_arity()[0]
    # random initialization of weights
    ex.set_weights([gdual([np.random.normal(0, 1)]) for _ in range(r * c) for _ in range(a)])
    # get active weights
    aw = [(an, i) for an in ex.get_active_nodes() if an >= n for i in range(a)]
    if len(aw) < 2:
        return
    for _ in range(100):
        # random choice of weights (that possibly minimize the error)
        num_vars = np.random.randint(2, min(3, len(aw)) + 1)  # number of weights (2 or 3)
        awidx = np.random.choice(len(aw), num_vars, replace=False)  # indices of chosen weights
        ani = [aw[j] for j in awidx]
        idx = [(an - n) * a + i for (an, i) in ani]
        ss = [f'w{an}_{i}' for (an, i) in ani]  # symbols
        w = ex.get_weights()  # initial weights
        for i in range(num_vars):
            w[idx[i]] = gdual(w[idx[i]].constant_cf, ss[i], 2)
        ex.set_weights(w)
        # compute the error
        E = get_diff(ex)
        Ei = sum(E.constant_cf)
        # get gradient and Hessian
        dw = np.zeros(num_vars)
        H = np.zeros((num_vars, num_vars))
        for k in range(num_vars):
            ssk = ss[k]
            dw[k] = collapse_vectorized_coefficient(E.get_derivative({'d'+ssk: 1}), n)
            H[k][k] = collapse_vectorized_coefficient(E.get_derivative({'d'+ssk: 2}), n)
            for l in range(k):
                H[l][k] = H[k][l] = collapse_vectorized_coefficient(E.get_derivative({'d'+ssk: 1, 'd'+ss[l]: 1}), n)
        # compute the updates
        try:
            updates = np.linalg.solve(H, -dw)
        except np.linalg.LinAlgError:
            continue
        # update the weights
        [ex.set_weight(ani[i][0], ani[i][1], w[idx[i]] + updates[i]) for i in range(num_vars)]
        wfe = ex.get_weights()
        for i in idx:
            wfe[i] = gdual(wfe[i].constant_cf)
        ex.set_weights(wfe)
        # if error increased, restore the initial weights
        Ef = sum(get_diff(ex).constant_cf)
        if not Ef < Ei:
            for i in idx:
                w[i] = gdual(w[i].constant_cf)
            ex.set_weights(w)


def run_experiment(dcgp: expression, screen_output=False) -> Tuple[int, expression, float]:
    """Runs an evolutionary strategy ES(1 + offspring)."""
    chromosome = [1] * offsprings
    fitness = [1] * offsprings
    weights = [1] * offsprings
    best_chromosome = dcgp.get()
    best_fitness = sum(get_diff(dcgp).constant_cf)
    if np.isnan(best_fitness):
        best_fitness = 1
    best_weights = dcgp.get_weights()
    for g in range(stop):
        for i in range(offsprings):
            dcgp.set(best_chromosome)
            dcgp.set_weights(best_weights)
            dcgp.mutate_active(i + 1)  # we mutate an increasingly higher number of active genes
            newton(dcgp)
            chromosome[i] = dcgp.get()
            fitness[i] = sum(get_diff(dcgp).constant_cf)
            weights[i] = dcgp.get_weights()
        for i in range(offsprings):
            if fitness[i] <= best_fitness:
                if (fitness[i] != best_fitness) and screen_output:
                    print(f'New best found: gen: {g}, value: {fitness[i]}')
                best_chromosome = chromosome[i]
                best_fitness = fitness[i]
                best_weights = weights[i]
                dcgp.set(best_chromosome)
                dcgp.set_weights(best_weights)
        if drop[g] < best_fitness or best_fitness < fitness_lower:
            break
    return g, best_chromosome, best_fitness


def run(i: int) -> None:
    if i % 100 == 0:
        print('.', end='\n' if i % 10000 == 9999 else '')
    dcgp = expression(inputs=3, outputs=1, rows=1, cols=6, levels_back=7, arity=2, kernels=kernels)
    for j in range(dcgp.get_n(), dcgp.get_n() + dcgp.get_rows() * dcgp.get_cols()):
        for k in range(dcgp.get_arity()[0]):
            dcgp.set_weight(j, k, gdual([np.random.normal(0, 1)]))
    g, best_chromosome, fitness = run_experiment(dcgp)
    dcgp.set(best_chromosome)
    if fitness < fitness_lower:
        print(f"\n{i}\t\t\t{g}\t{dcgp.simplify(['r', 'phi', 't'], True)}", end='')
        print(f'  with {dcgp.eph_symb} = {dcgp.eph_val}' if dcgp.eph_val else '')
        print(f' \t {fitness}')


def main() -> None:
    with Pool(32) as p:
        p.map(run, list(range(nexp)))


def get_error(u) -> float:
    """
    We provide a function for evaluating the error of an expression in an alternative way using SymPy.
    """
    ura = np.tile(1 / (4 * np.pi * tpoints) * np.exp(- ra ** 2 / (4 * tpoints)), nphi)
    urb = np.tile(1 / (4 * np.pi * tpoints) * np.exp(- rb ** 2 / (4 * tpoints)), nphi)
    uta = 1 / (4 * np.pi * ta) * np.exp(- rpoints ** 2 / (4 * ta)).repeat(nphi)
    utb = 1 / (4 * np.pi * tb) * np.exp(- rpoints ** 2 / (4 * tb)).repeat(nphi)
    diff_eq = np.array([(diff(u, t, 1) - (
                         diff(u, r, 2) + 1 / r * diff(u, r, 1) + 1 / r ** 2 * diff(u, phi, 2)
                         )).evalf(subs={r: i, phi: j, t: k}) for i in rpoints for j in phipoints for k in tpoints])
    diff_ra = np.array([u.evalf(subs={r: ra, phi: i, t: j}) for i in phipoints for j in tpoints]) - ura
    diff_rb = np.array([u.evalf(subs={r: rb, phi: i, t: j}) for i in phipoints for j in tpoints]) - urb
    diff_ta = np.array([u.evalf(subs={r: i, phi: j, t: ta}) for i in rpoints for j in phipoints]) - uta
    diff_tb = np.array([u.evalf(subs={r: i, phi: j, t: tb}) for i in rpoints for j in phipoints]) - utb
    return (np.sum(diff_eq * diff_eq) / n
            + (np.sum(diff_ra * diff_ra) + np.sum(diff_rb * diff_rb)) / (2 * nphi * nt)
            + (np.sum(diff_ta * diff_ta) + np.sum(diff_tb * diff_tb)) / (2 * nr * nphi))


def get_error_exact() -> float:
    u = 1 / (4 * np.pi * t) * exp(-r ** 2 / (4 * t))
    return get_error(u)


def get_error_dcgp() -> float:
    """
    10649			5	[0.0795774715459477*exp(-0.25*r**2/t)/t]
     3.244406406507316e-33
    or
    3317			124	[7.47236056879915e-19*r + 0.0795774715459477*exp(-0.25*r**2/t)/t]
     7.026328096821572e-33
    or
    23118			80	[0.0795774715459477*exp(-0.25*r**2/t)/t]
     1.4107816674966088e-32
    14641			21	[0.0795774715459477*exp(-0.25*r**2/t)/t]
     5.020553563966871e-32
    with kernel set +-*/, exp, 8 cols, 9 levels back.
    """
    u = 0.0795774715459477*exp(-0.25*r**2/t)/t
    return get_error(u)


if __name__ == '__main__':
    main()
    # print(f'error exact solution: {get_error_exact()}')
    # print(f'error dcgp solution: {get_error_dcgp()}')
