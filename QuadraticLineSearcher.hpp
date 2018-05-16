#ifndef RKTK_QUADRATIC_LINE_SEARCHER_HPP_INCLUDED
#define RKTK_QUADRATIC_LINE_SEARCHER_HPP_INCLUDED

// GNU MPFR multiprecision library headers
#include <mpfr.h>

// Project-specific headers
#include "MPFRVector.hpp"

namespace rktk {

    class QuadraticLineSearcher {

    private: // =============================================== MEMBER VARIABLES

        mpfr_prec_t precision;
        mpfr_t step_size, next_step_size;
        const mpfr_t &f0;
        mpfr_t f1, f2;
        mpfr_t numer, denom;

        mpfr_t &best_objective_value;
        mpfr_t &best_step_size;

        const MPFRVector &x0;
        const MPFRVector &dx;

    public: // ===================================================== CONSTRUCTOR

        QuadraticLineSearcher(mpfr_t &final_objective_value,
                              mpfr_t &final_step_size,
                              const MPFRVector &initial_point,
                              const mpfr_t &initial_objective_value,
                              const MPFRVector &step_direction,
                              mpfr_prec_t prec)
                : precision(prec), f0(initial_objective_value),
                  best_objective_value(final_objective_value),
                  best_step_size(final_step_size),
                  x0(initial_point), dx(step_direction) {
            mpfr_inits2(precision,
                        step_size, next_step_size,
                        f1, f2,
                        numer, denom,
                        best_objective_value, best_step_size,
                        static_cast<mpfr_ptr>(nullptr));
            mpfr_set(best_objective_value, initial_objective_value, MPFR_RNDN);
            mpfr_set_zero(best_step_size, 0);
        }

    public: // =========================================================== BIG 3

        QuadraticLineSearcher(const QuadraticLineSearcher &) = delete;

        QuadraticLineSearcher &operator=(
                const QuadraticLineSearcher &) = delete;

        ~QuadraticLineSearcher() {
            mpfr_clears(step_size, next_step_size,
                        f1, f2,
                        numer, denom,
                        static_cast<mpfr_ptr>(nullptr));
        }

    private: // ===================================== LINE SEARCH HELPER METHODS

        void evaluate_objective_function(mpfr_t objective_value,
                                         rktk::MPFRVector &xt,
                                         mpfr_t t, mpfr_rnd_t rnd,
                                         bool *changed = nullptr) {
            xt.set_axpy(t, dx, x0, rnd);
            if (x0 == xt) {
                if (changed != nullptr) { *changed = false; }
                mpfr_set(objective_value, f0, rnd);
                return;
            } else {
                if (changed != nullptr) { *changed = true; }
            }
            objective_function(objective_value, xt.data(), precision, rnd);
            if (mpfr_less_p(objective_value, best_objective_value)) {
                mpfr_set(best_objective_value, objective_value, rnd);
                mpfr_set(best_step_size, t, rnd);
            }
        }

    public: // ============================================= LINE SEARCH METHODS

        void search(rktk::MPFRVector &temp,
                    mpfr_t initial_step_size, mpfr_rnd_t rnd) {
            mpfr_set(step_size, initial_step_size, rnd);
            evaluate_objective_function(f1, temp, step_size, rnd);
            if (mpfr_less_p(f1, f0)) {
                int num_increases = 0;
                while (true) {
                    mpfr_mul_2ui(next_step_size, step_size, 1, rnd);
                    evaluate_objective_function(f2, temp, next_step_size, rnd);
                    if (mpfr_greaterequal_p(f2, f1)) {
                        break;
                    } else {
                        mpfr_swap(step_size, next_step_size);
                        mpfr_swap(f1, f2);
                        ++num_increases;
                        if (num_increases >= 4) {
                            return;
                        }
                    }
                }
                mpfr_mul_2ui(denom, f1, 1, rnd); // denom = 2*f1
                mpfr_sub(denom, denom, f2, rnd); // denom = 2*f1 - f2
                mpfr_sub(denom, denom, f0, rnd); // denom = 2*f1 - f2 - f0
                mpfr_mul_2ui(numer, f1, 2, rnd); // numer = 4*f1
                mpfr_sub(numer, numer, f2, rnd); // numer = 4*f1 - f2
                mpfr_mul_ui(f1, f0, 3, rnd);    // temporarily store f1' = 3*f0
                mpfr_sub(numer, numer, f1, rnd); // numer = 4*f1 - f2 - 3*f
                mpfr_div_2ui(next_step_size, step_size, 1, rnd);
                mpfr_mul(next_step_size, next_step_size, numer, rnd);
                mpfr_div(next_step_size, next_step_size, denom, rnd);
                evaluate_objective_function(f2, temp, next_step_size, rnd);
            } else {
                while (true) {
                    mpfr_div_2ui(next_step_size, step_size, 1, rnd);
                    bool changed;
                    evaluate_objective_function(f2, temp, next_step_size, rnd,
                                                &changed);
                    if (!changed) { return; }
                    if (mpfr_less_p(f2, f0)) {
                        break;
                    } else {
                        mpfr_swap(step_size, next_step_size);
                        mpfr_swap(f1, f2);
                    }
                }
                mpfr_mul_2ui(f2, f2, 1, rnd);
                mpfr_sub(denom, f1, f2, rnd);
                mpfr_add(denom, denom, f0, rnd);
                mpfr_mul_2ui(f2, f2, 1, rnd);
                mpfr_sub(numer, f1, f2, rnd);
                mpfr_mul_ui(f1, f0, 3, rnd);
                mpfr_add(numer, numer, f1, rnd);
                mpfr_div_2ui(next_step_size, step_size, 2, rnd);
                mpfr_mul(next_step_size, next_step_size, numer, rnd);
                mpfr_div(next_step_size, next_step_size, denom, rnd);
                evaluate_objective_function(f2, temp, next_step_size, rnd);
            }
        }

    }; // class QuadraticLineSearcher

} // namespace rktk

#endif // RKTK_QUADRATIC_LINE_SEARCHER_HPP_INCLUDED
