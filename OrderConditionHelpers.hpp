#ifndef RKTK_ORDER_CONDITION_HELPERS_HPP_INCLUDED
#define RKTK_ORDER_CONDITION_HELPERS_HPP_INCLUDED

// C++ standard library headers
#include <cstddef> // for std::size_t

// GNU MPFR multiprecision library headers
#include <mpfr.h>

// Project-specific headers
#include "MPFRVector.hpp"

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
          mpfr_t *mat, mpfr_rnd_t rnd) {
    std::size_t k = 0;
    for (std::size_t i = 0; i < dst_size; ++i) {
        mpfr_set(dst[i], mat[k], rnd);
        ++k;
        for (std::size_t j = 0; j < i; ++j, ++k) {
            mpfr_add(dst[i], dst[i], mat[k], rnd);
        }
    }
}

void lrss(mpfr_t *dst_re, mpfr_t *dst_du,
          std::size_t n,
          mpfr_t *mat_re, std::size_t mat_di, mpfr_rnd_t rnd) {
    lrsm(dst_re, n, mat_re, rnd);
    // TODO: See lsrq above.
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        mpfr_set_si(dst_du[i], k == mat_di, rnd);
        ++k;
        for (std::size_t j = 0; j < i; ++j, ++k) {
            mpfr_add_si(dst_du[i], dst_du[i], k == mat_di, rnd);
        }
    }
}

// =============================================================================

void elmm(mpfr_t *dst,
          std::size_t n,
          mpfr_t *v, mpfr_t *w, mpfr_rnd_t rnd) {
    for (std::size_t i = 0; i < n; ++i) { mpfr_mul(dst[i], v[i], w[i], rnd); }
}

void elmz(mpfr_t *dst_re, mpfr_t *dst_du,
          std::size_t n,
          mpfr_t *v_re, mpfr_t *v_du,
          mpfr_t *w_re, mpfr_t *w_du, mpfr_rnd_t rnd) {
    elmm(dst_re, n, v_re, w_re, rnd);
    for (std::size_t i = 0; i < n; ++i) {
        // TODO: Use mpfr_fmma here when we upgrade to MPFR 4.
        mpfr_mul(dst_du[i], v_du[i], w_re[i], rnd);
        mpfr_fma(dst_du[i], v_re[i], w_du[i], dst_du[i], rnd);
    }
}

// =============================================================================

void esqm(mpfr_t *dst,
          std::size_t n,
          mpfr_t *v, mpfr_rnd_t rnd) {
    for (std::size_t i = 0; i < n; ++i) { mpfr_sqr(dst[i], v[i], rnd); }
}

void esqz(mpfr_t *dst_re, mpfr_t *dst_du,
          std::size_t n,
          mpfr_t *v_re, mpfr_t *v_du, mpfr_rnd_t rnd) {
    for (std::size_t i = 0; i < n; ++i) {
        mpfr_mul(dst_du[i], v_re[i], v_du[i], rnd);
        mpfr_mul_2ui(dst_du[i], dst_du[i], 1, rnd);
        mpfr_sqr(dst_re[i], v_re[i], rnd);
    }
}

// =============================================================================

void dotm(mpfr_t dst,
          std::size_t n,
          mpfr_t *v, mpfr_t *w, mpfr_rnd_t rnd) {
    mpfr_mul(dst, v[0], w[0], rnd);
    for (std::size_t i = 1; i < n; ++i) {
        mpfr_fma(dst, v[i], w[i], dst, rnd);
    }
}

// =============================================================================

void lvmm(mpfr_t *dst,
          std::size_t dst_size, std::size_t mat_size,
          mpfr_t *mat, mpfr_t *vec, mpfr_rnd_t rnd) {
    std::size_t skp = mat_size - dst_size;
    std::size_t idx = skp * (skp + 1) / 2 - 1;
    for (std::size_t i = 0; i < dst_size; ++i, idx += skp, ++skp) {
        dotm(dst[i], i + 1, mat + idx, vec, rnd);
    }
}

void lvms(mpfr_t *dst_re, mpfr_t *dst_du,
          std::size_t dst_size, std::size_t mat_size,
          mpfr_t *mat_re, std::size_t mat_di,
          mpfr_t *vec_re, mpfr_t *vec_du, mpfr_rnd_t rnd) {
    lvmm(dst_re, dst_size, mat_size, mat_re, vec_re, rnd);
    std::size_t skp = mat_size - dst_size;
    std::size_t idx = skp * (skp + 1) / 2 - 1;
    for (std::size_t i = 0; i < dst_size; ++i, idx += skp, ++skp) {
        dotm(dst_du[i], i + 1, mat_re + idx, vec_du, rnd);
        if (idx <= mat_di && mat_di <= idx + i) {
            mpfr_add(dst_du[i], dst_du[i], vec_re[mat_di - idx], rnd);
        }
    }
}

// =============================================================================

void srim(mpfr_t dst, unsigned long int src, mpfr_rnd_t rnd) {
    mpfr_set_ui(dst, src, rnd);
    mpfr_ui_div(dst, 1, dst, rnd);
}

// =============================================================================

void ress(mpfr_t dst, mpfr_t tmp_re, mpfr_t tmp_du,
          std::size_t n,
          mpfr_t *m_re, mpfr_t *m_du, std::size_t m_offset,
          mpfr_t *x_re, std::size_t x_di, std::size_t x_offset,
          mpfr_t gamma, mpfr_rnd_t rnd) {
    dotm(tmp_re, n, m_re + m_offset, x_re + x_offset, rnd);
    mpfr_sub(tmp_re, tmp_re, gamma, rnd);
    mpfr_mul_2ui(tmp_re, tmp_re, 1, rnd);
    dotm(tmp_du, n, m_du + m_offset, x_re + x_offset, rnd);
    if (x_offset <= x_di && x_di < x_offset + n) {
        mpfr_add(tmp_du, tmp_du, m_re[m_offset + (x_di - x_offset)], rnd);
    }
    mpfr_fma(dst, tmp_re, tmp_du, dst, rnd);
}

void resm(mpfr_t f, mpfr_t tmp,
          std::size_t n,
          mpfr_t *m, mpfr_t *x, mpfr_t gamma, mpfr_rnd_t rnd) {
    dotm(tmp, n, m, x, rnd);
    mpfr_sub(tmp, tmp, gamma, rnd);
    mpfr_fma(f, tmp, tmp, f, rnd);
}

// =============================================================================

#endif // RKTK_ORDER_CONDITION_HELPERS_HPP_INCLUDED
