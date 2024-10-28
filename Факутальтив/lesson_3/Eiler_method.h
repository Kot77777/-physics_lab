#ifndef EILER_METHOD_H
#define EILER_METHOD_H
#include <array>
#include <vector>
#include <functional>
#include "States.h"

template<typename T, std::size_t N>
State<T, N> step_Eiler(const State<T, N>& cond, const double dt, std::function<T(const State<T, N>&)> f) {
    std::array<T, N> cond_i;
    for (std::size_t i = 0; i < N - 1; ++i) {
        cond_i[i] = cond[i] + dt * cond[i + 1];
    }

    cond_i[N - 1] = cond[N-1] + dt * f(cond);
    double time_next = cond.time + dt;
    return State<T, N>{cond_i, time_next};
}

template<typename T, std::size_t N>
State_full<T, N> solut(std::function<T(const State<T, N>&)> f, const double end_time, const double dt, const State<T, N>& cond_0) {
    std::vector<State<T, N>> solution;
    solution.push_back(cond_0);

    while (solution.back().time < end_time) {
        solution.push_back(step_Eiler(solution.back(), dt, f));
    }

    return State_full<T, N>{solution};
}

#endif //EILER_METHOD_H
