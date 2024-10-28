#include <iostream>
#include <array>
#include <vector>
#include <fstream>
#include "Eiler_method.h"
#include "Eiler_method_naiv.h"
#include "R-K_method.h"
#include "Heun_method.h"
#include "/home/kostya/Documents/json/include/nlohmann/json.hpp"
#include <chrono>

using namespace std::chrono;

int main()
{
    const double end_time = 2000;
    const double dt = 0.1;
    State<double, 2> cond_0{{1, 0}, 0.1};
    State_naiv cond_0_naiv{{1, 0}, 0};

    auto equation = [](const State<double, 2>& y) {
        return -sin(y[0]) + y[1];
    };

    auto start = high_resolution_clock::now();
    State_full<double, 2> s = solut<double, 2>(equation, end_time, dt, cond_0);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "abstract timing: " << duration.count() << std::endl;

    start = high_resolution_clock::now();
    State_full_naiv s_naiv = solut_Eiler(end_time, dt, cond_0_naiv, 1);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    std::cout << "naiv algorithm timing: " << duration.count() << std::endl;

    std::ifstream configFile("config.json");

    nlohmann::json config;
    configFile >> config;

    std::ofstream outFile("output.txt");
    for (int i = 0; i < s.solution.size(); ++i) {
        outFile << s_naiv.solution[i].cond[0] << " " << s_naiv.solution[i].cond[1] << '\n';
    }
    outFile.close();


    /*std::ifstream configFile("config.json");

    nlohmann::json config;
    configFile >> config;

    const double x_0 = config["x_0"];
    const double v_0 = config["v_0"];
    const double start_time = config["start_time"];
    const double dt = config["dt"];
    const double end_time = config["end_time"];
    const double w = config["w"];

    const State state_0{{x_0, v_0}, start_time};
    const State_R_K state_0_R_K{{x_0, v_0}, start_time};
    const State_Heun state_0_Heun{{x_0, v_0}, start_time};

    State_full s = solut(end_time, dt, state_0, w);
    State_full_R_K s_R_K = solut_R_K(end_time, dt, state_0_R_K, w);
    State_full_Heun s_Heun = solut_Heun(end_time, dt, state_0_Heun, w);

    std::ofstream outFile("output.txt");
    for (int i = 0; i < s.solution.size(); ++i) {
        outFile << s.solution[i].cond[0] << " " << s.solution[i].cond[1] << '\n';
    }
    outFile.close();

    std::ofstream outFile_R_K("output_R_K.txt");
    for (int i = 0; i < s_R_K.solution.size(); ++i) {
        outFile_R_K << s_R_K.solution[i].cond[0] << " " << s_R_K.solution[i].cond[1] << '\n';
    }
    outFile.close();

    std::ofstream outFile_Heun("output_Heun.txt");
    for (int i = 0; i < s_Heun.solution.size(); ++i) {
        outFile_Heun << s_Heun.solution[i].cond[0] << " " << s_Heun.solution[i].cond[1] << '\n';
    }
    outFile.close();*/
}