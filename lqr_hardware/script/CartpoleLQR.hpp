#ifndef CARTPOLE_LQR_HPP
#define CARTPOLE_LQR_HPP

#include <torch/torch.h>
#include <torch/script.h>

#include <deque>
#include <vector>
#include <chrono>
#include <string>
#include <serial/serial.h>
#include "actuator.hpp"
#include "Logger.hpp"


class CartpoleLQR {
public: 
    explicit CartpoleLQR(Logger& logger);
    void run();

private:
    Logger& logger_;

    torch::jit::script::Module actuator_net_;
    torch::Tensor states_, torque_, K_, u_, ref_, error_, x_des_, q_des_;
    torch::Tensor sg_coeffs, sg_coeffs_rev;
    std::deque<torch::Tensor> angle_buffer_;
    std::deque<float> err_hist_, vel_hist_;
    std::vector<float> data;
    std::string CTRLmode;
    bool add_noise_;
    float noise_std_;

    int window_size, num_states, max_itr, counter, num_frame_stacks;
    float looptime; 
    float posCheck, torqueCheck;
    float KP_, KD_;
    float current_resp_act, current_resp_loop;
    double R;
    size_t bytesWaiting;
    torch::Tensor error_position, error_position_angle;

    std::chrono::duration<double> encoder_duration;
    std::chrono::high_resolution_clock::time_point t_pre, t_end, run_start, iter_start, iter_end;
   
    torch::Tensor compute_states();
    bool act(const torch::Tensor& x_des_cmd);
    bool safetyChecker();
    void init_buffers();
    void init_lqr();
    void init_filter();
    std::pair<torch::Tensor, torch::Tensor> get_state_encoder();
    void load_actuator_net();

};

#endif



