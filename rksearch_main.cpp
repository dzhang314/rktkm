// C++ standard library headers
#include <cmath>    // for std::isfinite
#include <cstdlib>  // for std::exit
#include <cstring>  // for std::strlen
#include <ctime>    // for std::clock
#include <iostream> // for std::cout

// RKTK headers
#include "nonlinear_optimizers.hpp" // for BFGSOptimizer

mpfr_prec_t get_precision(int argc, char **argv) {
    if (argc >= 2) {
        char *end;
        const long long precision = std::strtoll(argv[1], &end, 10);
        const bool read_whole_arg = (std::strlen(argv[1]) ==
                static_cast<std::size_t>(end - argv[1]));
        const bool is_positive = (precision > 0);
        const bool in_range = (precision <= INT_MAX);
        if (read_whole_arg && is_positive && in_range) {
            return static_cast<mpfr_prec_t>(precision);
        }
    }
    return 53;
}

double get_print_period(int argc, char **argv) {
    if (argc >= 3) {
        char *end;
        const double print_period = std::strtod(argv[2], &end);
        const bool read_whole_arg = (std::strlen(argv[2]) ==
                static_cast<std::size_t>(end - argv[2]));
        const bool is_finite = std::isfinite(print_period);
        const bool is_positive = (print_period >= 0.0);
        if (read_whole_arg && is_finite && is_positive) {
            return print_period;
        }
    }
    return 0.5;
}

int get_print_precision(int argc, char **argv) {
    if (argc >= 4) {
        char *end;
        const long long print_precision = std::strtoll(argv[3], &end, 10);
        const bool read_whole_arg = (std::strlen(argv[3]) ==
                static_cast<std::size_t>(end - argv[3]));
        const bool is_non_negative = (print_precision >= 0);
        const bool in_range = (print_precision <= INT_MAX);
        if (read_whole_arg && is_non_negative && in_range) {
            return static_cast<int>(print_precision);
        }
    }
    return 0;
}

enum class SearchMode {
    EXPLORE, REFINE
};

int main(int argc, char **argv) {
    const auto clocks_between_prints = static_cast<std::clock_t>(
            get_print_period(argc, argv) * CLOCKS_PER_SEC);
    std::clock_t last_print_clock;
    const mpfr_prec_t prec = get_precision(argc, argv);
    const int print_prec = get_print_precision(argc, argv);
    const SearchMode mode = (argc >= 5)
                            ? SearchMode::REFINE
                            : SearchMode::EXPLORE;
    BFGSOptimizer optimizer(NUM_VARS, prec, MPFR_RNDN);
    if (mode == SearchMode::REFINE) {
        optimizer.initialize_from_file(std::string(argv[4]));
    } else {
        optimizer.initialize_random();
    }
    optimizer.print(print_prec);
    optimizer.write_to_file();
    last_print_clock = std::clock();
    optimizer.set_step_size();
    while (true) {
        optimizer.step(print_prec);
        if (!optimizer.objective_function_has_decreased()) {
            optimizer.print(print_prec);
            std::cout << "Located candidate local minimum." << std::endl;
            optimizer.write_to_file();
            return EXIT_SUCCESS;
        }
        optimizer.shift();
        if (optimizer.get_iteration_count() % 100 == 0) {
            optimizer.write_to_file();
        }
        const std::clock_t current_clock = std::clock();
        if (current_clock - last_print_clock >= clocks_between_prints) {
            optimizer.print(print_prec);
            last_print_clock = current_clock;
        }
    }
}
