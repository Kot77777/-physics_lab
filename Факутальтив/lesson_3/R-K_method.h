#ifndef R_K_METHOD_H
#define R_K_METHOD_H
#include "/home/kostya/Repositories/RocketC-/project/eigen/Eigen/Dense"

struct State_R_K
{
    Eigen::Vector2d cond;
    double time;
};

struct State_full_R_K {
    std::vector<State_R_K> solution;
};

Eigen::Vector2d derivative_cond(const Eigen::Vector2d& cond, const double w) {
    const double acceleration = -w * w * cond[0];
    const Eigen::Vector2d dirave = {cond[1], acceleration};
    return dirave;
}

State_R_K step_R_K(const State_R_K& state_i, const double step, const double w) {

    const Eigen::Vector2d k1 = derivative_cond(state_i.cond,                 w);
    const Eigen::Vector2d k2 = derivative_cond(state_i.cond + step * k1 / 2, w);
    const Eigen::Vector2d k3 = derivative_cond(state_i.cond + step * k2 / 2, w);
    const Eigen::Vector2d k4 = derivative_cond(state_i.cond + step * k3,     w);

    const Eigen::Vector2d update_cond = state_i.cond + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
    const double updatedTime = state_i.time + step;

    return {update_cond, updatedTime};

}

State_full_R_K solut_R_K(const double end_time, const double dt, const State_R_K& cond_0, const double w)
{
    std::vector<State_R_K> solution;
    State_R_K cond_i = step_R_K(cond_0, dt, w);
    solution.push_back(cond_i);
    while(solution.back().time < end_time) {
        solution.push_back(step_R_K(solution.back(), dt, w));
    }
    return {solution};
}

#endif //R_K_METHOD_H
