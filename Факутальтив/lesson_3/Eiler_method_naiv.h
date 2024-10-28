#ifndef EILER_METHOD_NAIV_H
#define EILER_METHOD_NAIV_H
#include "/home/kostya/Repositories/RocketC-/project/eigen/Eigen/Dense"

struct State_naiv {
    Eigen::Vector2d cond;
    double time;
};

struct State_full_naiv {
    std::vector<State_naiv> solution;
};

State_naiv step_Eiler(const State_naiv& cond, const double dt, const double w) {
    const double x_next = cond.cond[0] + dt * cond.cond[1];
    const double v_next = cond.cond[1] - dt * (w * w * cond.cond[0]);
    const double time_next = cond.time + dt;

    return {{x_next, v_next}, time_next};
}

State_full_naiv solut_Eiler(const double end_time, const double dt, const State_naiv& cond_0, const double w) {
    std::vector<State_naiv> solution;
    State_naiv cond_i = step_Eiler(cond_0, dt, w);
    solution.push_back(cond_i);
    while(solution.back().time < end_time) {
        solution.push_back(step_Eiler(solution.back(), dt, w));
    }
    return State_full_naiv{solution};
}

#endif //EILER_METHOD_NAIV_H
