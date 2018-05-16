#ifndef RKTK_BFGS_SUBROUTINES_HPP_INCLUDED
#define RKTK_BFGS_SUBROUTINES_HPP_INCLUDED

// C++ standard library headers
#include <cstddef> // for std::size_t

// GNU MPFR library header
#include <mpfr.h>

// Project-specific headers
#include "MPFRVector.hpp"

void update_inverse_hessian(rktk::MPFRMatrix &inv_hess, std::size_t n,
                            const rktk::MPFRVector &delta_gradient,
                            mpfr_t step_size,
                            const rktk::MPFRVector &step_direction,
                            mpfr_prec_t prec, mpfr_rnd_t rnd);

#endif // RKTK_BFGS_SUBROUTINES_HPP_INCLUDED
