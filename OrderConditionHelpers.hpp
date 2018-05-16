#ifndef RKTK_ORDER_CONDITION_HELPERS_HPP_INCLUDED
#define RKTK_ORDER_CONDITION_HELPERS_HPP_INCLUDED

// C++ standard library headers
#include <cstddef> // for std::size_t

// GNU MPFR library headers
#include <mpfr.h>

/*
 * The following functions are helper subroutines used in the calculation
 * of Runge-Kutta order conditions. Their names are deliberately short and
 * unreadable because they are intended to be called thousands of times
 * in machine-generated code. Brevity keeps generated file sizes down.
 *
 * Each function has a four-character name, where the first three characters
 * indicate the mathematical operation the function performs, and the last
 * character indicates the data type on which the function operates. The
 * three-character mnemonic operation codes are listed below.
 *
 *     lrs - Lower-triangular matrix Row Sums
 *     elm - ELementwise Multiplication
 *     esq - Elementwise SQuare
 *     dot - DOT product
 *     lvm - Lower-triangular Matrix-Vector multiplication
 *     sqr - Scalar sQuaRe
 *     sri - Set to Reciprocal of (unsigned) Integer
 *     res - RESult (special operation used to evaluate partial
 *                   derivatives of Runge-Kutta order conditions)
 *
 * The one-character data type codes are listed below. Currently, only
 * double-precision and arbitrary-precision objective functions have been
 * implemented; other letters are reserved below for possible future expansion.
 *
 *     m - arbitrary precision (mpfr_t)
 *     s - indexed arbitrary precision (mpfr_t)
 *     z - dual arbitrary precision (mpfr_t)
 */

// =============================================================================

void lrsm(mpfr_t *dst,
          std::size_t dst_size,
          mpfr_t *mat, mpfr_rnd_t rnd);

void lrss(mpfr_t *dst_re, mpfr_t *dst_du,
          std::size_t n,
          mpfr_t *mat_re, std::size_t mat_di, mpfr_rnd_t rnd);

// =============================================================================

void elmm(mpfr_t *dst,
          std::size_t n,
          mpfr_t *v, mpfr_t *w, mpfr_rnd_t rnd);

void elmz(mpfr_t *dst_re, mpfr_t *dst_du,
          std::size_t n,
          mpfr_t *v_re, mpfr_t *v_du,
          mpfr_t *w_re, mpfr_t *w_du, mpfr_rnd_t rnd);

// =============================================================================

void esqm(mpfr_t *dst,
          std::size_t n,
          mpfr_t *v, mpfr_rnd_t rnd);

void esqz(mpfr_t *dst_re, mpfr_t *dst_du,
          std::size_t n,
          mpfr_t *v_re, mpfr_t *v_du, mpfr_rnd_t rnd);

// =============================================================================

void dotm(mpfr_t dst,
          std::size_t n,
          mpfr_t *v, mpfr_t *w, mpfr_rnd_t rnd);

// =============================================================================

void lvmm(mpfr_t *dst,
          std::size_t dst_size, std::size_t mat_size,
          mpfr_t *mat, mpfr_t *vec, mpfr_rnd_t rnd);

void lvms(mpfr_t *dst_re, mpfr_t *dst_du,
          std::size_t dst_size, std::size_t mat_size,
          mpfr_t *mat_re, std::size_t mat_di,
          mpfr_t *vec_re, mpfr_t *vec_du, mpfr_rnd_t rnd);

// =============================================================================

void srim(mpfr_t dst, unsigned long int src, mpfr_rnd_t rnd);

// =============================================================================

void ress(mpfr_t dst, mpfr_t tmp_re, mpfr_t tmp_du,
          std::size_t n,
          mpfr_t *m_re, mpfr_t *m_du, std::size_t m_offset,
          mpfr_t *x_re, std::size_t x_di, std::size_t x_offset,
          mpfr_t gamma, mpfr_rnd_t rnd);

void resm(mpfr_t f, mpfr_t tmp,
          std::size_t n,
          mpfr_t *m, mpfr_t *x, mpfr_t gamma, mpfr_rnd_t rnd);

// =============================================================================

#endif // RKTK_ORDER_CONDITION_HELPERS_HPP_INCLUDED
