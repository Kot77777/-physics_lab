#ifndef STATES_H
#define STATES_H
#include <array>
#include <vector>

template<typename T, std::size_t N>
struct State {
    std::array<T, N> cond;
    double time;

    T& operator[](const int i){return cond[i];}
    const T& operator[](const int i) const {return cond[i];}
};

template<typename T, std::size_t N>
State<T, N> operator+(const State<T, N>& state1, const State<T, N>& state2) {
    State<T, N> state_0;
    for (int i = 0; i < N; ++i) {
        state_0[i] = state1[i] + state2[i];
    }
}

template<typename T, std::size_t N>
struct State_full {
    std::vector<State<T, N>> solution;
};

#endif //STATES_H
