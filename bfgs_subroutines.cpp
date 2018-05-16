#include "bfgs_subroutines.hpp"

// RKTK headers
#include "objective_function.hpp" // for objective_function

static inline void dot(mpfr_t dst,
                       const rktk::MPFRVector &v, const rktk::MPFRVector &w,
                       mpfr_rnd_t rnd) {
    mpfr_mul(dst, v[0], w[0], rnd);
    for (std::size_t i = 1; i < NUM_VARS; ++i) {
        mpfr_fma(dst, v[i], w[i], dst, rnd);
    }
}

void update_inverse_hessian(rktk::MPFRMatrix &inv_hess, std::size_t n,
                            const rktk::MPFRVector &delta_gradient,
                            mpfr_t step_size,
                            const rktk::MPFRVector &step_direction,
                            mpfr_prec_t prec, mpfr_rnd_t rnd) {
    static rktk::MPFRVector *kappa = nullptr;
    static mpfr_t theta, lambda, sigma, beta, alpha;
    if (kappa == nullptr) {
        kappa = new rktk::MPFRVector(prec);
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
    for (std::size_t i = 0; i < n; ++i) {
        mpfr_fms((*kappa)[i], beta, step_direction[i], (*kappa)[i], rnd);
        mpfr_neg((*kappa)[i], (*kappa)[i], rnd);
    }
    mpfr_div(alpha, step_size, lambda, rnd);
    mpfr_neg(alpha, alpha, rnd);
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j, ++k) {
            mpfr_mul(beta, (*kappa)[i], step_direction[j], rnd);
            mpfr_fma(beta, step_direction[i], (*kappa)[j], beta, rnd);
            mpfr_fma(inv_hess.data()[k], alpha, beta, inv_hess.data()[k], rnd);
        }
    }
}
