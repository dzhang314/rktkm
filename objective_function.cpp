#include "objective_function.hpp"

#include "OrderConditionHelpers.hpp"
#include "RK_10_16.ipp"

void objective_gradient(mpfr_t *dst, std::size_t n,
                        mpfr_t *x, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    for (std::size_t i = 0; i < n; ++i) {
        objective_function_partial(dst[i], x, i, prec, rnd);
    }
}
