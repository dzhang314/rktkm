#ifndef RKTK_NONLINEAR_OPTIMIZERS_HPP
#define RKTK_NONLINEAR_OPTIMIZERS_HPP

// C++ standard library headers
#include <algorithm>  // for std::generate
#include <cstdlib>    // for std::exit
#include <fstream>    // for std::ifstream, std::ofstream
#include <functional> // for std::ref
#include <iomanip>    // for std::setw, std::setfill
#include <iostream>   // for std::cout
#include <iterator>   // for std::begin, std::end
#include <limits>     // for std::numeric_limits
#include <random>     // for std::uniform_real_distribution et al.
#include <utility>    // for std::swap

// RKTK headers
#include "objective_function.hpp"
#include "bfgs_subroutines.hpp"
#include "FilenameHelpers.hpp"
#include "MPFRMatrix.hpp"
#include "MPFRVector.hpp"

static inline void nan_check(const char *msg) {
    if (mpfr_nanflag_p()) {
        std::cout << "INTERNAL ERROR: Invalid calculation performed "
                  << msg << "." << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

class BFGSOptimizer {

    typedef std::numeric_limits<std::uint64_t> uint64_limits;

private: // ======================================================= DATA MEMBERS

    const mpfr_prec_t prec;
    const mpfr_rnd_t rnd;

    rktk::MPFRVector x;
    rktk::MPFRVector x_new;
    mpfr_t x_norm;
    mpfr_t x_new_norm;

    mpfr_t func;
    mpfr_t func_new;

    rktk::MPFRVector grad;
    rktk::MPFRVector grad_new;
    mpfr_t grad_norm;
    mpfr_t grad_new_norm;
    rktk::MPFRVector grad_delta;

    mpfr_t step_size;
    mpfr_t step_size_new;
    rktk::MPFRVector step_dir;

    rktk::MPFRMatrix hess_inv;

    std::size_t iter_count = std::numeric_limits<std::size_t>::max();

    std::uint64_t uuid_seg0 = uint64_limits::max();
    std::uint64_t uuid_seg1 = uint64_limits::max();
    std::uint64_t uuid_seg2 = uint64_limits::max();
    std::uint64_t uuid_seg3 = uint64_limits::max();
    std::uint64_t uuid_seg4 = uint64_limits::max();

public: // ======================================================== CONSTRUCTORS

    explicit BFGSOptimizer(mpfr_prec_t numeric_precision,
                           mpfr_rnd_t rounding_mode) :
            prec(numeric_precision), rnd(rounding_mode),
            x(prec), x_new(prec), grad(prec), grad_new(prec),
            grad_delta(prec), step_dir(prec), hess_inv(prec) {
        mpfr_init2(x_norm, prec);
        mpfr_init2(x_new_norm, prec);
        mpfr_init2(func, prec);
        mpfr_init2(func_new, prec);
        mpfr_init2(grad_norm, prec);
        mpfr_init2(grad_new_norm, prec);
        mpfr_init2(step_size, prec);
        mpfr_init2(step_size_new, prec);
    }

    // explicitly disallow copy construction
    BFGSOptimizer(const BFGSOptimizer &) = delete;

    // explicitly disallow copy assignment
    BFGSOptimizer &operator=(const BFGSOptimizer &) = delete;

public: // ========================================================== DESTRUCTOR

    ~BFGSOptimizer() {
        mpfr_clear(x_norm);
        mpfr_clear(x_new_norm);
        mpfr_clear(func);
        mpfr_clear(func_new);
        mpfr_clear(grad_norm);
        mpfr_clear(grad_new_norm);
        mpfr_clear(step_size);
        mpfr_clear(step_size_new);
    }

public: // ======================================================== INITIALIZERS

    void initialize_random() {
        nan_check("before workspace initialization");
        std::uint64_t seed[std::mt19937_64::state_size];
        std::random_device seed_source;
        std::generate(std::begin(seed), std::end(seed), std::ref(seed_source));
        std::seed_seq seed_sequence(std::begin(seed), std::end(seed));
        std::mt19937_64 random_engine(seed_sequence);
        std::uniform_real_distribution<long double> unif(0.0L, 1.0L);
        for (std::size_t i = 0; i < NUM_VARS; ++i) {
            mpfr_set_ld(x[i], unif(random_engine), rnd);
        }
        x.norm(x_norm, rnd);
        objective_function(func, x.data(), prec, rnd);
        objective_gradient(grad.data(), x.data(), prec, rnd);
        grad.norm(grad_norm, rnd);
        mpfr_set_zero(step_size, 0);
        hess_inv.set_identity_matrix();
        iter_count = 0;
        uuid_seg0 = random_engine() & 0xFFFFFFFF;
        uuid_seg1 = random_engine() & 0xFFFF;
        uuid_seg2 = random_engine() & 0xFFFF;
        uuid_seg3 = random_engine() & 0xFFFF;
        uuid_seg4 = random_engine() & 0xFFFFFFFFFFFF;
        nan_check("after workspace initialization");
    }

    void initialize_from_file(const std::string &filename) {
        std::cout << "Opening input file '" << filename << "'..." << std::endl;
        std::FILE *input_file = std::fopen(filename.c_str(), "r");
        if (input_file == nullptr) {
            std::cout << "ERROR: could not open input file." << std::endl;
            std::exit(EXIT_FAILURE);
        }
        std::cout << "Successfully opened input file. Reading..." << std::endl;
        for (std::size_t i = 0; i < NUM_VARS; ++i) {
            if (mpfr_inp_str(x[i], input_file, 10, rnd) == 0) {
                std::cout << "ERROR: Could not read input file entry at index "
                          << i << "." << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
        std::fclose(input_file);
        std::cout << "Successfully read input file." << std::endl;
        x.norm(x_norm, rnd);
        objective_function(func, x.data(), prec, rnd);
        objective_gradient(grad.data(), x.data(), prec, rnd);
        grad.norm(grad_norm, rnd);
        mpfr_set_zero(step_size, 0);
        hess_inv.set_identity_matrix();
        if (is_rktk_filename(filename)) {
            iter_count = dec_substr_to_int(filename, 52, 64);
            uuid_seg0 = hex_substr_to_int(filename, 15, 23);
            uuid_seg1 = hex_substr_to_int(filename, 24, 28);
            uuid_seg2 = hex_substr_to_int(filename, 29, 33);
            uuid_seg3 = hex_substr_to_int(filename, 34, 38);
            uuid_seg4 = hex_substr_to_int(filename, 39, 51);
        } else {
            iter_count = 0;
            std::uint64_t seed[std::mt19937_64::state_size];
            std::random_device seed_source;
            std::generate(std::begin(seed), std::end(seed),
                          std::ref(seed_source));
            std::seed_seq seed_sequence(std::begin(seed), std::end(seed));
            std::mt19937_64 random_engine(seed_sequence);
            uuid_seg0 = random_engine() & 0xFFFFFFFF;
            uuid_seg1 = random_engine() & 0xFFFF;
            uuid_seg2 = random_engine() & 0xFFFF;
            uuid_seg3 = random_engine() & 0xFFFF;
            uuid_seg4 = random_engine() & 0xFFFFFFFFFFFF;
        }
    }

public: // =========================================================== ACCESSORS

    std::size_t get_iteration_count() { return iter_count; }

    bool objective_function_has_decreased() {
        return (mpfr_less_p(func_new, func) != 0);
    }

    void print(int print_precision) {
        if (print_precision <= 0) {
            const long double log102 = 0.301029995663981195213738894724493027L;
            print_precision = static_cast<int>(
                    static_cast<long double>(prec) * log102);
            print_precision += 2;
        }
        mpfr_printf(
                "%012zu | %+.*RNe | %+.*RNe | %+.*RNe | %+.*RNe\n", iter_count,
                print_precision, func, print_precision, grad_norm,
                print_precision, step_size, print_precision, x_norm);
    }

    void write_to_file() {
        const long double log102 = 0.301029995663981195213738894724493027L;
        const int print_precision =
                static_cast<int>(static_cast<long double>(prec) * log102) + 2;
        auto f_score = static_cast<int>(
                -100.0L * std::log10(mpfr_get_ld(func, rnd)));
        if (f_score < 0) { f_score = 0; }
        if (f_score > 9999) { f_score = 9999; }
        auto g_score = static_cast<int>(
                -100.0L * std::log10(mpfr_get_ld(grad_norm, rnd)));
        if (g_score < 0) { g_score = 0; }
        if (g_score > 9999) { g_score = 9999; }
        std::ostringstream filename;
        filename << std::setfill('0') << std::dec;
        filename << std::setw(4) << f_score << '-';
        filename << std::setw(4) << g_score << "-RKTK-";
        filename << std::hex << std::uppercase;
        filename << std::setw(8) << uuid_seg0 << '-';
        filename << std::setw(4) << uuid_seg1 << '-';
        filename << std::setw(4) << uuid_seg2 << '-';
        filename << std::setw(4) << uuid_seg3 << '-';
        filename << std::setw(12) << uuid_seg4 << '-';
        filename << std::dec;
        filename << std::setw(12) << iter_count << ".txt";
        std::string filename_string(filename.str());
        std::FILE *output_file = std::fopen(filename_string.c_str(), "w+");
        for (std::size_t i = 0; i < NUM_VARS; ++i) {
            mpfr_fprintf(output_file, "%+.*RNe\n", print_precision, x[i]);
        }
        mpfr_fprintf(output_file, "\n");
        mpfr_fprintf(output_file, "Objective function value: %+.*RNe\n",
                     print_precision, func);
        mpfr_fprintf(output_file, "Objective gradient norm:  %+.*RNe\n",
                     print_precision, grad_norm);
        mpfr_fprintf(output_file, "Most recent step size:    %+.*RNe\n",
                     print_precision, step_size);
        mpfr_fprintf(output_file, "Distance from origin:     %+.*RNe\n",
                     print_precision, x_norm);
        std::fclose(output_file);
    }

public: // ============================================================ MUTATORS

    void set_step_size() {
        mpfr_set_ui(step_size, 1, rnd);
        mpfr_div_2ui(step_size, step_size,
                     static_cast<unsigned long>(prec / 2), rnd);
    }

    void step(int print_precision) {
        nan_check("before performing BFGS iteration");
        // Compute a quasi-Newton step direction by multiplying the approximate
        // inverse Hessian matrix by the gradient vector. Negate the result to
        // obtain a direction of local decrease (rather than increase).
        step_dir.set_matrix_vector_multiply(hess_inv, grad, rnd);
        nan_check("during calculation of BFGS step direction");
        // Normalize the step direction to ensure consistency of step sizes.
        step_dir.negate_and_normalize(func_new, rnd);
        nan_check("during normalization of BFGS step direction");
        // Compute a near-optimal step size via quadratic line search.
        quadratic_line_search(step_size_new, x_new, grad_new,
                              x, func, step_size, step_dir, prec, rnd);
        nan_check("during quadratic line search");
        after_line_search:
        if (mpfr_zero_p(step_size_new)) {
            print(print_precision);
            std::cout << "NOTICE: Optimal step size reduced to zero. "
                    "Resetting approximate inverse Hessian matrix to the "
                    "identity matrix and re-trying line search." << std::endl;
            hess_inv.set_identity_matrix();
            step_dir.set_matrix_vector_multiply(hess_inv, grad, rnd);
            step_dir.negate_and_normalize(func_new, rnd);
            quadratic_line_search(step_size_new, x_new, grad_new,
                                  x, func, step_size, step_dir, prec, rnd);
            if (mpfr_zero_p(step_size_new)) {
                std::cout << "NOTICE: Optimal step size reduced to zero again "
                        "after Hessian reset. BFGS iteration has converged to "
                        "the requested precision." << std::endl;
                x_new = x;
                mpfr_set(x_new_norm, x_norm, rnd);
                mpfr_set(func_new, func, rnd);
                grad_new = grad;
                mpfr_set(grad_new_norm, grad_norm, rnd);
                grad_delta.set_zero();
                return;
            }
        }
        // Take a step using the computed step direction and step size.
        for (std::size_t i = 0; i < NUM_VARS; ++i) {
            mpfr_fma(x_new[i], step_size_new, step_dir[i], x[i], rnd);
        }
        nan_check("while taking BFGS step");
        x_new.norm(x_new_norm, rnd);
        nan_check("while evaluating norm of new point");
        // Evaluate the objective function at the new point.
        objective_function(func_new, x_new.data(), prec, rnd);
        nan_check("during evaluation of objective function at new point");
        // Ensure that the objective function has decreased.
        if (!objective_function_has_decreased()) {
            print(print_precision);
            if (mpfr_less_p(step_size_new, step_size)) {
                std::cout << "NOTICE: BFGS step failed to decrease "
                        "objective function value after decreasing "
                        "step size. Re-trying line search with smaller "
                        "initial step size." << std::endl;
                mpfr_swap(step_size, step_size_new);
                quadratic_line_search(step_size_new, x_new, grad_new,
                                      x, func, step_size, step_dir, prec, rnd);
                nan_check("during quadratic line search");
            } else {
                std::cout << "NOTICE: BFGS step failed to decrease objective "
                        "function after increasing step size. Reverting "
                        "to smaller original step size." << std::endl;
                mpfr_set(step_size_new, step_size, rnd);
            }
            goto after_line_search;
        }
        // Evaluate the gradient vector at the new point.
        objective_gradient(grad_new.data(), x_new.data(), prec, rnd);
        nan_check("during evaluation of objective gradient at new point");
        grad_new.norm(grad_new_norm, rnd);
        nan_check("while evaluating norm of objective gradient");
        // Use difference between previous and current gradient vectors to
        // perform a rank-one update of the approximate inverse Hessian matrix.
        grad_delta.set_sub(grad_new, grad, rnd);
        nan_check("while subtracting consecutive gradient vectors");
        update_inverse_hessian(hess_inv, NUM_VARS,
                               grad_delta, step_size_new, step_dir, prec, rnd);
        nan_check("while updating approximate inverse Hessian");
    }

    void shift() {
        x.swap(x_new);
        mpfr_set(x_norm, x_new_norm, rnd);
        mpfr_set(func, func_new, rnd);
        grad.swap(grad_new);
        mpfr_set(grad_norm, grad_new_norm, rnd);
        mpfr_set(step_size, step_size_new, rnd);
        ++iter_count;
    }

};

#endif // RKTK_NONLINEAR_OPTIMIZERS_HPP
