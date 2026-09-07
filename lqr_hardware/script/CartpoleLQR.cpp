#include "CartpoleLQR.hpp"

#include <torch/torch.h>
#include <torch/script.h>

#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <thread>
#include <filesystem>
#include <string>
#include <cmath>
#include <serial/serial.h>
#include "actuator.hpp"
#include "Logger.hpp"

extern serial::Serial arduino; 
extern Actuator motor;         
extern float resp_0; 

#define MOTOR_ID 0x07

CartpoleLQR::CartpoleLQR(Logger& logger)
    : logger_(logger),
      num_states(4),        // Number of one stack of states
      max_itr(100000),      // Iterations of main control loop
      R(0.025490),          // D pully = 49.79mm in cataluge
      looptime(2.),
      posCheck(0.3),
      torqueCheck(9.0),
      CTRLmode("LQR"), // switch between LQR ,LQRPD and LQRACTUATORNET, note that you should use "" not ''
      add_noise_(true),
      noise_std_(0.5),
      num_frame_stacks(21),
      KP_(10.0),
      KD_(1.0),
      counter(0) {
    init_filter();
    init_lqr();
    init_buffers();
    load_actuator_net();
}

void CartpoleLQR::run() {
    run_start = std::chrono::high_resolution_clock::now();
    t_pre = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < max_itr; ++i) {
        iter_start = std::chrono::high_resolution_clock::now();
        // cout << "counter: " << counter << std::endl;
        
        states_ = compute_states();

        if (CTRLmode == "LQR"){
            error_ = states_ - ref_;
            u_ = - torch::mm(K_, error_);
            torque_ = torch::clamp(u_ * R, -5.0f, 5.0f);
        } else if (CTRLmode == "LQRPD"){
            error_ = states_ - ref_;
            x_des_ = - torch::mm(K_, error_);

            // add noise for data collecting 
            // if (add_noise_){
            //     auto noise = torch::randn_like(x_des_) * noise_std_;
            //     x_des_ = x_des_ + noise;
            // }
            
            // for loging
            q_des_ = x_des_ / R;
            error_position = x_des_ - states_[0][0];
            error_position_angle = q_des_ - (motor.cycle_responses[1] - resp_0);
            // 
            u_ = KP_ * (x_des_ - states_[0][0]) - KD_ * states_[1][0];
            torque_ = torch::clamp(u_ * R, -5.0f, 5.0f);
        } else if (CTRLmode == "LQRACTUATORNET"){
            error_ = states_ - ref_;
            x_des_ = - torch::mm(K_, error_);
            
            // for loging
            q_des_ = x_des_ / R;
            error_position = x_des_ - states_[0][0];
            error_position_angle = q_des_ - (motor.cycle_responses[1] - resp_0);
            //
            err_hist_.push_back((x_des_ - states_[0][0]).item<float>());
            vel_hist_.push_back(states_[1][0].item<float>());
            err_hist_.pop_front();
            vel_hist_.pop_front();

            float e0  = err_hist_[num_frame_stacks - 1];          // current
            float v0  = vel_hist_[num_frame_stacks - 1];
            float e10 = err_hist_[num_frame_stacks - 1 - 10];     // 10 steps ago
            float v10 = vel_hist_[num_frame_stacks - 1 - 10];
            float e20 = err_hist_[num_frame_stacks - 1 - 20];     // 20 steps ago
            float v20 = vel_hist_[num_frame_stacks - 1 - 20];
            
            torch::Tensor input = torch::tensor({e0, v0, e10, v10, e20, v20}, torch::kFloat32).unsqueeze(0);
            torque_ = actuator_net_.forward({input}).toTensor();
            torque_.clamp_(-5.0f, 5.0f); 
        } else {
            std::cerr << "Unknown Control mode: " << CTRLmode << std::endl;
            break;
        }
        
        auto lqr_end = std::chrono::high_resolution_clock::now();
        
        if(act(torque_)){
            std::cout << "Breaking loop due to safety trigger." << std::endl;
            break;
        }
        // cout << "Torque output of the policy: " << torque_.item<float>() << std::endl;
        
        auto act_end = std::chrono::high_resolution_clock::now();

        // log states and control input
        float x_cart = states_[0][0].item<float>();
        float vel_cart = states_[1][0].item<float>();
        float ang_pole = states_[2][0].item<float>();
        float vel_pole = states_[3][0].item<float>();
        float control_u =  u_.item<float>();
        float x_cart_des =  x_des_.item<float>();
        float torque_input = torque_.item<float>();
        float error_position_pd = error_position.item<float>();
        float error_position_angle_pd = error_position_angle.item<float>();

        auto middle_end = std::chrono::high_resolution_clock::now();

        // dynamic delay time
        t_end = std::chrono::high_resolution_clock::now();
        while(std::chrono::duration<double, std::milli>(t_end-t_pre).count() < looptime){
            t_end = std::chrono::high_resolution_clock::now();
        }
        t_pre = std::chrono::high_resolution_clock::now();
        
        current_resp_loop = motor.cycle_responses[3];

        iter_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> whole_loop_duration = iter_end - iter_start;
        std::chrono::duration<double> time = iter_end - run_start;
        
        // log time
        float time_sec = time.count();
        float encoder_duration_s = encoder_duration.count();
        float iteration_duration_s = whole_loop_duration.count();
        
        data = {time_sec, x_cart, vel_cart, ang_pole, vel_pole, torque_input, x_cart_des, 
                iteration_duration_s, static_cast<float>(bytesWaiting),
                current_resp_act, current_resp_loop, error_position_pd, error_position_angle_pd};
                
        logger_.logData(data);

        counter ++;
    }
}

torch::Tensor CartpoleLQR::compute_states() {
    auto encoder_start = std::chrono::high_resolution_clock::now();
    auto [angle_rad, angle_vel] = get_state_encoder();
    auto encoder_end = std::chrono::high_resolution_clock::now();
    encoder_duration = encoder_end - encoder_start;

    states_[0][0] = (motor.cycle_responses[1] - resp_0) * R;
    states_[1][0] = (motor.cycle_responses[2] * R);
    states_[2][0] = - angle_rad.item<float>();
    states_[3][0] = - angle_vel.item<float>();

    return states_;
}

bool CartpoleLQR::act(const torch::Tensor& torque_cmd) {
    if (safetyChecker()) return true;

    motor.command(MOTOR_ID, 0, 0, 0, 0, torque_cmd.item<double>());
    current_resp_act = motor.cycle_responses[3];

    if (safetyChecker()) return true;

    return false;
}

bool CartpoleLQR::safetyChecker() {
    if (std::abs(motor.cycle_responses[3]) > torqueCheck) {
        motor.disable(MOTOR_ID);
        std::cout << "----- torque safety triggered!! -----" << std::endl;
        return true;
    }

    if (std::abs((motor.cycle_responses[1] - resp_0) * R) > posCheck) {
        motor.disable(MOTOR_ID);
        std::cout << "x cart: " << (motor.cycle_responses[1] - resp_0) * R << std::endl;
        std::cout << "----- x cart safety triggered!! -----" << std::endl;
        return true;
    }

    return false;
}

void CartpoleLQR::init_buffers() {
    for (int i = 0; i < window_size; ++i) {
            angle_buffer_.push_back(torch::zeros({1}, torch::kFloat32));
        }
    states_ = torch::zeros({num_states, 1}, torch::kFloat32);
    torque_ = torch::zeros({1, 1}, torch::kFloat32);
    
    
    for (int i = 0; i < num_frame_stacks; ++i) {
        err_hist_.push_back(0.0f);
        vel_hist_.push_back(0.0f);
    }
}

void CartpoleLQR::init_lqr() {

    if (CTRLmode == "LQR"){
        K_ = torch::tensor({{-350.5691, -300.4080, -1000.3087, -170.7064}}, torch::kFloat32);
    } else if (CTRLmode == "LQRPD"){
        // K_ = torch::tensor({{-60.9964,  -48.5124, -256.7047,  -30.81133}}, torch::kFloat32);
        K_ = torch::tensor({{-12.05,  -8.85124, -50.67047,  -6.081133}}, torch::kFloat32);
    }

    ref_ = torch::tensor({{0.0f}, {0.0f}, {0.0f}, {0.0f}}, torch::kFloat32); 
    u_ = torch::zeros({1, 1}, torch::kFloat32);
    x_des_ = torch::zeros({1, 1}, torch::kFloat32);
    q_des_ = torch::zeros({1, 1}, torch::kFloat32);
    error_position = torch::zeros({1, 1}, torch::kFloat32);
    error_position_angle = torch::zeros({1, 1}, torch::kFloat32);
}

void CartpoleLQR::init_filter() {
    window_size = 13;
    sg_coeffs = torch::tensor({
    -23.5805861f, 13.7362637f, 32.8421578f, 37.3792874f,
    30.9898435f, 17.3160173f, 0.0f, -17.3160173f,
    -30.9898435f, -37.3792874f, -32.8421578f, -13.7362637f,
    23.5805861f
    }, torch::kFloat32);
    sg_coeffs_rev = sg_coeffs.flip(0);
}

std::pair<torch::Tensor, torch::Tensor> CartpoleLQR::get_state_encoder() {
        
    bytesWaiting = arduino.available();

    std::string line;
    while(arduino.available()){
        line = arduino.readline();
    }

    float angle_deg_val = 0.0f; // encoder angle degree

    try {
        angle_deg_val = std::stof(line);
    } catch (const std::exception& e) {
        std::cerr << "invalid line: " << line << " — " << e.what() << std::endl;
        angle_deg_val = angle_buffer_.back().item<float>() * 180.0 / M_PI;

    }
    
    torch::Tensor angle_deg = torch::tensor({angle_deg_val}, torch::kFloat32);
    torch::Tensor angle_rad = angle_deg * M_PI / 180;

    angle_buffer_.push_back(angle_rad);
    angle_buffer_.pop_front();

    torch::Tensor angular_velocity = torch::tensor({0.0f}, torch::kFloat32);

    if (angle_buffer_.size() == window_size) {
        for (int i = 0; i < window_size; ++i) {
            angular_velocity += sg_coeffs_rev[i] * angle_buffer_[i];
        }
    }
    
    if (torch::abs(angular_velocity).item<float>() < 1e-5) {
        angular_velocity = torch::zeros({1}, torch::kFloat32);
    }

    return {angle_rad, angular_velocity};
}

void CartpoleLQR::load_actuator_net() {
    // Building path to the JIT actuator net file
    std::string root = std::filesystem::path(__FILE__).parent_path().parent_path();
    std::string actuator_net_path = root + "/actuator_net/2025-09-28_15-50-51_/JIT_model.pt";
    // Loading JIT actuator net
    actuator_net_ = torch::jit::load(actuator_net_path);
    actuator_net_.eval();    
    std::cout<<"------ actuator net Loaded!!!! ------"<< std::endl;
    std::cout << "Loading jit model from: " << actuator_net_path << std::endl;
};