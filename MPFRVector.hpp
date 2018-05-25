#ifndef RKTK_MPFR_VECTOR_HPP_INCLUDED
#define RKTK_MPFR_VECTOR_HPP_INCLUDED

// C++ standard library headers
#include <cstddef> // for std::size_t
#include <utility> // for std::swap

// GNU MPFR multiprecision library headers
#include <mpfr.h>

// Project-specific headers
#include "MPFRMatrix.hpp"
#include "objective_function.hpp"

namespace rktk {

    class MPFRVector {

    private: // =============================================== MEMBER VARIABLES

        mpfr_prec_t precision;
        mpfr_t entries[NUM_VARS];

    public: // ===================================================== CONSTRUCTOR

        MPFRVector(mpfr_prec_t prec) : precision(prec) {
            for (mpfr_t &x : entries) { mpfr_init2(x, precision); }
        }

    public: // =========================================================== BIG 3

        MPFRVector(const MPFRVector &rhs) : precision(rhs.precision) {
            for (std::size_t i = 0; i < NUM_VARS; ++i) {
                mpfr_init2(entries[i], precision);
                mpfr_set(entries[i], rhs.entries[i], MPFR_RNDN);
            }
        }

        MPFRVector &operator=(const MPFRVector &rhs) {
            for (std::size_t i = 0; i < NUM_VARS; ++i) {
                mpfr_set(entries[i], rhs.entries[i], MPFR_RNDN);
            }
            return *this;
        }

        ~MPFRVector() {
            for (mpfr_t &x : entries) { mpfr_clear(x); }
        }

    public: // ======================================================== MUTATORS

        void swap(MPFRVector &rhs) {
            std::swap(precision, rhs.precision);
            std::swap(entries, rhs.entries);
        }

    public: // ============================================== INDEXING OPERATORS

        mpfr_t &operator[](std::size_t i) { return entries[i]; }

        const mpfr_t &operator[](std::size_t i) const { return entries[i]; }

        mpfr_t *data() { return entries; }

        const mpfr_t *data() const { return entries; }

    public: // =================================================================

        bool operator==(const MPFRVector &rhs) const {
            for (std::size_t i = 0; i < NUM_VARS; ++i) {
                if (!mpfr_equal_p(entries[i], rhs.entries[i])) { return false; }
            }
            return true;
        }

        void set_zero() {
            for (mpfr_t &x : entries) { mpfr_set_zero(x, 0); }
        }

        void scale(mpfr_srcptr coeff, mpfr_rnd_t rnd) {
            for (mpfr_t &x : entries) { mpfr_mul(x, coeff, x, rnd); }
        }

        void norm(mpfr_ptr dst, mpfr_rnd_t rnd) {
            mpfr_sqr(dst, entries[0], rnd);
            for (std::size_t i = 1; i < NUM_VARS; ++i) {
                mpfr_fma(dst, entries[i], entries[i], dst, rnd);
            }
            mpfr_sqrt(dst, dst, rnd);
        }

        void negate_and_normalize(mpfr_ptr tmp, mpfr_rnd_t rnd) {
            norm(tmp, rnd);
            mpfr_si_div(tmp, -1, tmp, rnd);
            scale(tmp, rnd);
        }

        void set_add(const MPFRVector &x, const MPFRVector &y, mpfr_rnd_t rnd) {
            for (std::size_t i = 0; i < NUM_VARS; ++i) {
                mpfr_add(entries[i], x.entries[i], y.entries[i], rnd);
            }
        }

        void set_sub(const MPFRVector &x, const MPFRVector &y, mpfr_rnd_t rnd) {
            for (std::size_t i = 0; i < NUM_VARS; ++i) {
                mpfr_sub(entries[i], x.entries[i], y.entries[i], rnd);
            }
        }

        void set_axpy(mpfr_t a, const MPFRVector &x, const MPFRVector &y,
                      mpfr_rnd_t rnd) {
            for (std::size_t i = 0; i < NUM_VARS; ++i) {
                mpfr_fma(entries[i], a, x.entries[i], y.entries[i], rnd);
            }
        }

        void set_axmy(mpfr_t a, const MPFRVector &x, const MPFRVector &y,
                      mpfr_rnd_t rnd) {
            for (std::size_t i = 0; i < NUM_VARS; ++i) {
                mpfr_fms(entries[i], a, x.entries[i], y.entries[i], rnd);
            }
        }

        void set_matrix_vector_multiply(
                const MPFRMatrix &mat, const MPFRVector &vec, mpfr_rnd_t rnd) {
            for (std::size_t i = 0, k = 0; i < NUM_VARS; ++i) {
                mpfr_mul(entries[i], mat.data()[k], vec[0], rnd);
                ++k;
                for (std::size_t j = 1; j < NUM_VARS; ++j, ++k) {
                    mpfr_fma(entries[i],
                             mat.data()[k], vec[j], entries[i], rnd);
                }
            }
        }

    }; // class MPFRVector

} // namespace rktk

#endif // RKTK_MPFR_VECTOR_HPP_INCLUDED
