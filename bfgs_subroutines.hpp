#ifndef RKTK_BFGS_SUBROUTINES_HPP_INCLUDED
#define RKTK_BFGS_SUBROUTINES_HPP_INCLUDED

// C++ standard library headers
#include <cstddef> // for std::size_t

// GNU MPFR library header
#include <mpfr.h>

void quadratic_line_search(mpfr_t optimal_step_size,
                           mpfr_t *temp1, mpfr_t *temp2, std::size_t n,
                           mpfr_t *x, mpfr_t f,
                           mpfr_t initial_step_size,
                           mpfr_t *step_direction,
                           mpfr_prec_t prec, mpfr_rnd_t rnd);

void update_inverse_hessian(mpfr_t *inv_hess, std::size_t n,
                            mpfr_t *delta_gradient,
                            mpfr_t step_size, mpfr_t *step_direction,
                            mpfr_prec_t prec, mpfr_rnd_t rnd);

#endif // RKTK_BFGS_SUBROUTINES_HPP_INCLUDED
