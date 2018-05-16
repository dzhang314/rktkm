#ifndef RKTK_MPFR_MATRIX_HPP_INCLUDED
#define RKTK_MPFR_MATRIX_HPP_INCLUDED

// C++ standard library headers
#include <cstddef> // for std::size_t
#include <utility> // for std::swap

// GNU MPFR multiprecision library headers
#include <mpfr.h>

// Project-specific headers
#include "objective_function.hpp"

namespace rktk {

    class MPFRMatrix {

    private: // =============================================== MEMBER VARIABLES

        mpfr_prec_t precision;
        mpfr_t entries[NUM_VARS * NUM_VARS];

    public: // ===================================================== CONSTRUCTOR

        MPFRMatrix(mpfr_prec_t prec) : precision(prec) {
            for (mpfr_t &x : entries) { mpfr_init2(x, precision); }
        }

    public: // =========================================================== BIG 3

        MPFRMatrix(const MPFRMatrix &) = delete;

        MPFRMatrix &operator=(const MPFRMatrix &) = delete;

        ~MPFRMatrix() {
            for (mpfr_t &x : entries) { mpfr_clear(x); }
        }

    public: // ============================================== INDEXING OPERATORS

        mpfr_t *data() { return entries; }

        const mpfr_t *data() const { return entries; }

    public: // =================================================================

        void set_identity_matrix() {
            for (std::size_t i = 0, k = 0; i < NUM_VARS; ++i) {
                for (std::size_t j = 0; j < NUM_VARS; ++j, ++k) {
                    mpfr_set_si(entries[k], i == j, MPFR_RNDN);
                }
            }
        }


    }; // class MPFRMatrix

} // namespace rktk

#endif // RKTK_MPFR_MATRIX_HPP_INCLUDED
