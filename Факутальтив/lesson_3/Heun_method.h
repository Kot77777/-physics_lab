#ifndef HEUN_METHOD_H
#define HEUN_METHOD_H
#include <vector>
#include "/home/kostya/Repositories/RocketC-/project/eigen/Eigen/Dense"

struct State_Heun {
    Eigen::Vector2d cond;
    double time;
};

struct State_full_Heun {
    std::vector<State_Heun> solution;
};

State_Heun step_Heun(const State_Heun& cond, const State_Heun& cond_pred, const double dt, const double w)
{
    const double x_next = cond_pred.cond[0] + dt * cond.cond[1];
    const double v_next = cond_pred.cond[1] - dt * (w * w * cond.cond[0]);
    return {{x_next, v_next}, cond.time};
}

State_Heun step_Eiler(const State_Heun& cond, const double dt, const double w) {
    const double x_next = cond.cond[0] + dt * cond.cond[1];
    const double v_next = cond.cond[1] - dt * (w * w * cond.cond[0]);
    const double time_next = cond.time + dt;
    const State_Heun Eiler_i = {{x_next, v_next}, time_next};
    const State_Heun for_heun = {0.5 * (Eiler_i.cond + cond.cond), time_next};
    return step_Heun(for_heun, cond, dt, w);
}

State_full_Heun solut_Heun(const double end_time, const double dt, const State_Heun& cond_0, const double w) {
    std::vector<State_Heun> solution;
    State_Heun cond_i = step_Eiler(cond_0, dt, w);
    solution.push_back(cond_i);
    while(solution.back().time < end_time) {
        solution.push_back(step_Eiler(solution.back(), dt, w));
    }
    return State_full_Heun{solution};
}

#endif //HEUN_METHOD_H
