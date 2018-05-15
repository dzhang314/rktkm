#ifndef RKTK_LINALG_SUBROUTINES_HPP_INCLUDED
#define RKTK_LINALG_SUBROUTINES_HPP_INCLUDED

// C++ standard library headers
#include <cstddef> // for std::size_t

// GNU MPFR library headers
#include <mpfr.h>

bool elementwise_equal(std::size_t n, mpfr_t *v, mpfr_t *w);

void identity_matrix(mpfr_t *mat, std::size_t n, mpfr_rnd_t rnd);

void l2_norm(mpfr_t dst, std::size_t n, mpfr_t *v, mpfr_rnd_t rnd);

void matrix_vector_multiply(mpfr_t *dst, std::size_t n,
                            mpfr_t *mat, mpfr_t *vec, mpfr_rnd_t rnd);

#endif // RKTK_LINALG_SUBROUTINES_HPP_INCLUDED
