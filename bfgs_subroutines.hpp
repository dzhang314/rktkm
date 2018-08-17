#ifndef RKTK_BFGS_SUBROUTINES_HPP_INCLUDED
#define RKTK_BFGS_SUBROUTINES_HPP_INCLUDED

// C++ standard library headers
#include <cstddef> // for std::size_t

// GNU MPFR library header
#include <mpfr.h>

// Project-specific headers
#include <dznl/MPFRVector.hpp>
#include "objective_function.hpp" // for objective_function

static inline void dot(mpfr_t dst,
                       const dznl::MPFRVector &v, const dznl::MPFRVector &w,
                       mpfr_rnd_t rnd) {
    mpfr_mul(dst, v[0], w[0], rnd);
    for (std::size_t i = 1; i < NUM_VARS; ++i) {
        mpfr_fma(dst, v[i], w[i], dst, rnd);
    }
}

void update_inverse_hessian(dznl::MPFRMatrix &inv_hess,
                            const dznl::MPFRVector &delta_gradient,
                            mpfr_t step_size,
                            const dznl::MPFRVector &step_direction,
                            mpfr_prec_t prec, mpfr_rnd_t rnd) {
    static dznl::MPFRVector *kappa = nullptr;
    static mpfr_t theta, lambda, sigma, beta, alpha;
    if (kappa == nullptr) {
        kappa = new dznl::MPFRVector(NUM_VARS, prec);
        mpfr_init2(theta, prec);
        mpfr_init2(lambda, prec);
        mpfr_init2(sigma, prec);
        mpfr_init2(beta, prec);
        mpfr_init2(alpha, prec);
    }
    // nan_check("during initialization of inverse hessian update workspace");
    kappa->set_matrix_vector_multiply(inv_hess, delta_gradient, rnd);
    // nan_check("during evaluation of kappa");
    dot(theta, delta_gradient, *kappa, rnd);
    // nan_check("during evaluation of theta");
    dot(lambda, delta_gradient, step_direction, rnd);
    mpfr_mul(lambda, lambda, step_size, rnd);
    // nan_check("during evaluation of lambda");
    mpfr_sqr(beta, lambda, rnd);
    mpfr_add(sigma, lambda, theta, rnd);
    mpfr_div(sigma, sigma, beta, rnd);
    // nan_check("during evaluation of sigma");
    mpfr_mul(beta, step_size, lambda, rnd);
    mpfr_mul(beta, beta, sigma, rnd);
    mpfr_div_2ui(beta, beta, 1, rnd);
    for (std::size_t i = 0; i < NUM_VARS; ++i) {
        mpfr_fms((*kappa)[i], beta, step_direction[i], (*kappa)[i], rnd);
        mpfr_neg((*kappa)[i], (*kappa)[i], rnd);
    }
    mpfr_div(alpha, step_size, lambda, rnd);
    mpfr_neg(alpha, alpha, rnd);
    for (std::size_t i = 0, k = 0; i < NUM_VARS; ++i) {
        for (std::size_t j = 0; j < NUM_VARS; ++j, ++k) {
            mpfr_mul(beta, (*kappa)[i], step_direction[j], rnd);
            mpfr_fma(beta, step_direction[i], (*kappa)[j], beta, rnd);
            mpfr_fma(inv_hess.data()[k], alpha, beta, inv_hess.data()[k], rnd);
        }
    }
}

void update_inverse_hessian_mbfgst(
        dznl::MPFRMatrix &inv_hess,
        mpfr_t func, mpfr_t func_new,
        const dznl::MPFRVector &grad, const dznl::MPFRVector &grad_new,
        const dznl::MPFRVector &delta_gradient,
        mpfr_t step_size,
        const dznl::MPFRVector &step_direction,
        mpfr_prec_t prec, mpfr_rnd_t rnd) {
    static dznl::MPFRVector *w = nullptr;
    static mpfr_t phi, phi_0, t0, t1, t2, t3, beta, rho;
    if (w == nullptr) {
        w = new dznl::MPFRVector(NUM_VARS, prec);
        mpfr_inits2(prec,
                    phi, phi_0, t0, t1, t2, t3, beta, rho,
                    static_cast<mpfr_ptr>(nullptr));
    }
    // nan_check("during initialization of inverse hessian update workspace");
    mpfr_sub(phi, func, func_new, rnd);
    mpfr_mul_2si(phi, phi, +2, rnd);
    // phi = 4 * (func - func_new);
    w->set_add(grad, grad_new, rnd);
    dot(phi_0, *w, step_direction, rnd);
    mpfr_mul(phi_0, step_size, phi_0, rnd);
    mpfr_mul_2si(phi_0, phi_0, +1, rnd);
    // phi_0 = 2 * step_size * dot(grad + grad_new, step_direction);
    mpfr_add(phi, phi, phi_0, rnd);
    // phi += phi_0; calculation of phi completed.
    w->set_matrix_vector_multiply(inv_hess, delta_gradient, rnd);
    dot(t0, step_direction, delta_gradient, rnd);
    mpfr_si_div(t1, +1, t0, rnd);
    dot(t2, delta_gradient, *w, rnd);
    mpfr_div(rho, t1, step_size, rnd);
    mpfr_mul(beta, phi, rho, rnd);
    mpfr_add_si(beta, beta, +1, rnd);
    mpfr_div(t3, step_size, beta, rnd);
    mpfr_fma(t3, t1, t2, t3, rnd);
    mpfr_div_2si(t3, t3, +1, rnd);
    w->set_axmy(t3, step_direction, *w, rnd);
    for (std::size_t i = 0, k = 0; i < NUM_VARS; ++i) {
        for (std::size_t j = 0; j < NUM_VARS; ++j, ++k) {
            mpfr_mul(t0, (*w)[i], step_direction[j], rnd);
            mpfr_fma(t0, step_direction[i], (*w)[j], t0, rnd);
            mpfr_fma(inv_hess.data()[k], t1, t0, inv_hess.data()[k], rnd);
        }
    }
}

#endif // RKTK_BFGS_SUBROUTINES_HPP_INCLUDED
