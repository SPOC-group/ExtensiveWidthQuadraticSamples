import sys
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root


# TODO: Add analytical edges for the MP + GOE case

def integral_qhat(qhat, kappa_val):
    # assert qhat > 0
    
    sigma = 1/np.sqrt(np.abs(qhat))
    
    def solve_poly(z, sigma, kappa):
        p = [sigma**2, -(z + kappa*sigma**2), 1 - kappa + kappa*z, -kappa]
        return np.roots(p)


    def rho(x):        
        return np.max(np.imag(solve_poly(x+1e-4j, sigma, kappa_val))) / np.pi
    

    def edges_rho(sigma, kappa):
        edges_poly = [kappa**2,
                -2*kappa - 2*kappa**2 - 2*kappa**3*sigma**2,
                1 - 2*kappa + kappa**2 - 2*kappa**2*sigma**2 + 8*kappa**3*sigma**2 + kappa**4*sigma**4,
                8*kappa*sigma**2 + 2*kappa**2*sigma**2 - 10*kappa**3*sigma**2 + 8*kappa**3*sigma**4 - 2*kappa**4*sigma**4,
                -4*sigma**2 + 12*kappa*sigma**2 - 12*kappa**2*sigma**2 + 4*kappa**3*sigma**2 - 8*kappa**2*sigma**4 - 20*kappa**3*sigma**4 + kappa**4*sigma**4 - 4*kappa**4*sigma**6]
        roots_all = np.roots(edges_poly)
        real_roots = np.real(roots_all[np.abs(np.imag(roots_all)) < 1e-3])

        return np.sort(real_roots)

    
    edges_list = edges_rho(sigma, kappa_val)

    if len(edges_list) == 4:
        return quad(lambda x: rho(x)**3, edges_list[0], edges_list[1], epsabs=1e-8, epsrel=1e-8)[0] + quad(lambda x: rho(x)**3, edges_list[2], edges_list[3], epsabs=1e-8, epsrel=1e-8)[0]
    else:
        return quad(lambda x: rho(x)**3, edges_list[0], edges_list[1], epsabs=1e-8, epsrel=1e-8)[0]



def q_hat_eq(q_hat, alpha, kappa):
    return integral_qhat(q_hat[0], kappa) / q_hat[0] - 3 * (1-2*alpha) / 4 / np.pi**2


def theory(alpha, kappa):
    # q_hat = root(q_hat_eq, x0=2*alpha, args=(alpha, kappa)).x[0]
    Q_0 = 1 + 1/kappa
    q_hat = root(q_hat_eq, x0=2*alpha/Q_0, args=(alpha, kappa)).x[0]

    # return q_hat
    # print(2*alpha/Q_0)
    # print(2*alpha/q_hat)

    # return q_hat
    return 2*alpha/q_hat


def main(kappa, alpha_min=1e-3):
    alpha = np.linspace(alpha_min, kappa - kappa**2/2 - 1e-5, 129)
    gen_error = np.zeros_like(alpha)

    for i,a in enumerate(alpha):
        gen_error[i] = theory(a, kappa)
        print(f"iter {i}: alpha = {a}, gen_error = {gen_error[i]}")

    # save q and alpha in one file
    np.save(f'data_theory/SE_k={kappa}.npy', {'alpha': alpha, 'gen_error': gen_error})


if __name__ == '__main__':
    kappa = float(sys.argv[1])
    main(kappa)