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

#include <dznl/MPFRMatrix.hpp>
#include <dznl/MPFRVector.hpp>
#include <dznl/MPFRQuadraticLineSearcher.hpp>

static inline void nan_check(const char *msg) {
    if (mpfr_nanflag_p()) {
        std::cout << "INTERNAL ERROR: Invalid calculation performed "
                  << msg << "." << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

enum class StepType {
    BFGS, GRAD, NONE
};

class BFGSOptimizer {

    typedef std::numeric_limits<std::uint64_t> uint64_limits;

private: // ======================================================= DATA MEMBERS

    const mpfr_prec_t prec;
    const mpfr_rnd_t rnd;
    StepType step_type;

    dznl::MPFRVector x, x_new, grad, grad_new, grad_delta;
    mpfr_t x_norm, x_new_norm, grad_norm, grad_new_norm;

    mpfr_t func, func_grad, func_new;
    mpfr_t step_size, step_size_grad, step_size_new;
    dznl::MPFRVector grad_dir, step_dir;

    dznl::MPFRMatrix hess_inv;

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
            step_type(StepType::NONE),
            x(NUM_VARS, prec), x_new(NUM_VARS, prec),
            grad(NUM_VARS, prec), grad_new(NUM_VARS, prec),
            grad_delta(NUM_VARS, prec), grad_dir(NUM_VARS, prec),
            step_dir(NUM_VARS, prec), hess_inv(NUM_VARS, prec) {
        mpfr_inits2(
                prec,
                x_norm, x_new_norm, grad_norm, grad_new_norm,
                func, func_grad, func_new,
                step_size, step_size_grad, step_size_new,
                static_cast<mpfr_ptr>(nullptr));
    }

    // explicitly disallow copy construction
    BFGSOptimizer(const BFGSOptimizer &) = delete;

    // explicitly disallow copy assignment
    BFGSOptimizer &operator=(const BFGSOptimizer &) = delete;

public: // ========================================================== DESTRUCTOR

    ~BFGSOptimizer() {
        mpfr_clears(
                x_norm, x_new_norm, grad_norm, grad_new_norm,
                func, func_grad, func_new,
                step_size, step_size_grad, step_size_new,
                static_cast<mpfr_ptr>(nullptr));
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
                "%012zu | %+.*RNe | %+.*RNe | %+.*RNe | %+.*RNe | ", iter_count,
                print_precision, func, print_precision, grad_norm,
                print_precision, step_size, print_precision, x_norm);
        switch (step_type) {
            case StepType::BFGS:
                std::cout << "BFGS" << std::endl;
                break;
            case StepType::GRAD:
                std::cout << "GRAD" << std::endl;
                break;
            case StepType::NONE:
                std::cout << "NONE" << std::endl;
                break;
        }
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
        grad_dir = grad;
        grad_dir.negate_and_normalize(func_new, rnd);
        step_dir.set_matrix_vector_multiply(hess_inv, grad, rnd);
        nan_check("during calculation of BFGS step direction");
        // Normalize the step direction to ensure consistency of step sizes.
        step_dir.negate_and_normalize(func_new, rnd);
        nan_check("during normalization of BFGS step direction");
        // Compute a near-optimal step size via quadratic line search.
        {
            dznl::MPFRQuadraticLineSearcher grad_searcher(
                    func_grad, step_size_grad,
                    objective_function, x, func, grad_dir, prec, rnd);
            grad_searcher.search(step_size);
            dznl::MPFRQuadraticLineSearcher bfgs_searcher(
                    func_new, step_size_new,
                    objective_function, x, func, step_dir, prec, rnd);
            bfgs_searcher.search(step_size);
            if (mpfr_less_p(func_grad, func_new)) {
                step_dir = grad_dir;
                hess_inv.set_identity_matrix();
                mpfr_set(func_new, func_grad, rnd);
                mpfr_set(step_size_new, step_size_grad, rnd);
                step_type = StepType::GRAD;
            } else {
                step_type = StepType::BFGS;
            }
        }
        nan_check("during quadratic line search");
        if (mpfr_zero_p(step_size_new)) {
            print(print_precision);
            std::cout << "NOTICE: Optimal step size reduced to zero. BFGS "
                         "iteration has converged to the requested precision."
                      << std::endl;
            return;
        }
        // Take a step using the computed step direction and step size.
        x_new.set_axpy(step_size_new, step_dir, x, rnd);
        x_new.norm(x_new_norm, rnd);
        objective_function(func_new, x_new.data(), prec, rnd);
        // Evaluate the gradient vector at the new point.
        objective_gradient(grad_new.data(), x_new.data(), prec, rnd);
        nan_check("during evaluation of objective gradient at new point");
        grad_new.norm(grad_new_norm, rnd);
        nan_check("while evaluating norm of objective gradient");
        // Use difference between previous and current gradient vectors to
        // perform a rank-one update of the approximate inverse Hessian matrix.
        grad_delta.set_sub(grad_new, grad, rnd);
        nan_check("while subtracting consecutive gradient vectors");
        update_inverse_hessian(hess_inv,
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
