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

namespace fs = std::filesystem;
using namespace std;
using namespace torch;

#define NUM_STATES 4            // Number of one stack of states
#define MAX_ITR 1000000         // Iterations of main control loop
#define MOTOR_ID 0x07           // 0x07
#define R 0.025490              // D pully = 49.79mm in cataluge

// Global 
serial::Serial arduino;            
Actuator motor("can0"); 
float resp_0 = 0;


class CartpoleSim2real {
public:
    CartpoleSim2real()
        : states_(torch::zeros({NUM_STATES, 1}, torch::kFloat32)),
          torque_(torch::zeros({1, 1}, torch::kFloat32)),
          K_(torch::tensor({{ 447.2136,  613.4864, 3469.5271,  100.0895}}, torch::kFloat32)),
          ref_(torch::tensor({{0.0f}, {0.0f}, {static_cast<float>(M_PI)}, {0.0f}}, torch::kFloat32)), 
          u_(torch::zeros({1, 1}, torch::kFloat32)),
          looptime(2.0),
          counter(0),
          iteration_duration_s(0.002),
          window_size(11),
          sg_coeffs(torch::tensor({
            -29.1375291f, 28.5547786f, 51.6705517f, 48.8539239f,
            28.7490287f, 0.0f, -28.7490287f, -48.8539239f,
            -51.6705517f, -28.5547786f, 29.1375291f
             }, torch::kFloat32)),
          sg_coeffs_rev(sg_coeffs.flip(0))

    {
        for (int i = 0; i < window_size; ++i) {
              angle_buffer_.push_back(torch::zeros({1}, torch::kFloat32));
          }
        createLogFile();
    }

    void run() {
        auto run_start = std::chrono::high_resolution_clock::now();
        states_ = compute_states();
        std::this_thread::sleep_for(chrono::microseconds(600));
        
        std::chrono::high_resolution_clock::time_point t_pre;
        std::chrono::high_resolution_clock::time_point t_end;

        t_pre = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < MAX_ITR; ++i) {
            auto iter_start = std::chrono::high_resolution_clock::now();
            // cout << "counter: " << counter << endl;
            
            error_ = states_ - ref_;
            u_ = - torch::mm(K_, error_);
            torque_ = torch::clamp(u_ * R, -12.0f, 12.0f);
            
            auto lqr_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> lqr_duration = lqr_end - iter_start;

            act(torque_);
            // cout << "Torque output of the policy: " << torque_.item<float>() << endl;
            
            auto act_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> act_duration = act_end - lqr_end;

            // log states and control input
            float x_cart = states_[0][0].item<float>();
            float vel_cart = states_[1][0].item<float>();
            float ang_pole = states_[2][0].item<float>();
            float vel_pole = states_[3][0].item<float>();
            float control_u =  u_.item<float>();
            float torque_input = torque_.item<float>();
            
            auto log_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> middle_log_duration = log_end - act_end;

            states_ = compute_states();

            auto middle_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> loop_duration = middle_end - iter_start;
            std::chrono::duration<double> compute_states_duration = middle_end - log_end;
            std::chrono::duration<double> time = middle_end - run_start;
            
            // log time
            float time_sec = time.count();
            float loop_time_s = loop_duration.count();
            float lqr_duration_s = lqr_duration.count();
            float act_duration_s = act_duration.count();
            float compute_states_duration_s = compute_states_duration.count();
            float encoder_duration_s = encoder_duration.count();

            if (log_file_.is_open()) {
                log_file_ << time_sec << ","
                          << x_cart << "," << vel_cart << ","
                          << ang_pole << "," << vel_pole << ","
                          << control_u << "," << torque_input << ","
                          << loop_time_s << "," << lqr_duration_s << ","
                          << act_duration_s << "," << compute_states_duration_s << "," 
                          << encoder_duration_s << ","  
                          << iteration_duration_s << "," << bytesWaiting << "\n"; 
                log_file_.flush();
            }
            
            counter ++;
            // dynamic delay time
            t_end = std::chrono::high_resolution_clock::now();
            while(std::chrono::duration<double, std::milli>(t_end-t_pre).count() < looptime){
                t_end = std::chrono::high_resolution_clock::now();
            }
            t_pre = std::chrono::high_resolution_clock::now();

            auto iter_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> whole_loop_duration = iter_end - iter_start;
            iteration_duration_s = whole_loop_duration.count();

        }
    }

private:
    Tensor states_;
    Tensor torque_;
    Tensor ref_;
    Tensor K_, u_, error_;
    float looptime;
    int counter;
    size_t bytesWaiting;
    ofstream log_file_;
    float iteration_duration_s;

    std::chrono::duration<double> encoder_duration;

    std::deque<torch::Tensor> angle_buffer_;
    torch::Tensor sg_coeffs;
    torch::Tensor sg_coeffs_rev;
    int window_size;


    torch::Tensor compute_states() {

        auto encoder_start = std::chrono::high_resolution_clock::now();
        auto [angle_rad, angle_vel] = get_state_encoder();
        auto encoder_end = std::chrono::high_resolution_clock::now();
        encoder_duration = encoder_end - encoder_start;

        states_[0][0] = (motor.cycle_responses[1] - resp_0) * R;
        states_[1][0] = (motor.cycle_responses[2] * R);
        states_[2][0] = angle_rad.item<float>() + static_cast<float>(M_PI);
        states_[3][0] = angle_vel.item<float>();

        return states_;
    }
    
    void act(Tensor torque_) {
        // torque = torch::zeros_like(torque_);
        if (std::abs((torque_).item<double>()) > 13){
                motor.disable(MOTOR_ID);
                cout<< "-----torque safty triggered!!-----" <<endl;
            }
        motor.command(MOTOR_ID, 0, 0, 0, 0, torque_.item<double>());

        if (std::abs((motor.cycle_responses[1] - resp_0) * R) > 0.21){ 
                motor.disable(MOTOR_ID);
                cout << "x cart in  act function: " << (motor.cycle_responses[1] - resp_0) * R << endl;
                cout << "-----x cart safty triggered!!-----" <<endl;
        } 
        // std::this_thread::sleep_for(chrono::microseconds(100)); // best choise is 100/900 , -/1100
    }
    
    void createLogFile() {
        // Create logs directory and timestamped subdirectory
        fs::path current_path = fs::current_path();  
        std::string base_dir = (current_path.parent_path() / "logs").string();         
        auto now = chrono::system_clock::now();
        std::time_t now_c = chrono::system_clock::to_time_t(now);

        std::stringstream folder_name_ss;
        folder_name_ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d_%H-%M-%S");
        std::string folder_name = folder_name_ss.str();

        fs::path full_dir = fs::path(base_dir) / folder_name;
        if (!fs::exists(full_dir)) {
            fs::create_directories(full_dir);
        }

        std::string log_file_path = (full_dir / "lqr_data.csv").string();
        log_file_.open(log_file_path);
        log_file_ << "time_sec,x_cart,vel_cart,ang_pole,vel_pole,"
                  << "torque_input,loop_time_s,lqr_duration_s,act_duration_s,"
                  << "compute_states_duration_s,encoder_duration_s,"
                  << "iteration_duration_s,bytesWaiting" << "\n";
    }

    // Function to read state from encoder (angle and angular velocity)
    pair<torch::Tensor, torch::Tensor> get_state_encoder() {
        
        bytesWaiting = arduino.available();

        string line;
        while(arduino.available()){
            line = arduino.readline();
        }

        float angle_deg_val = 0.0f; // encoder angle degree

        try {
            angle_deg_val = std::stof(line);
        } catch (const std::exception& e) {
            std::cerr << "invalide line: " << line << " — " << e.what() << std::endl;
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
            angular_velocity = torch::zeros({1}, torch::kFloat);
        }

        return {angle_rad, angular_velocity};
    }
    
};

int main() {

    auto enableStart = std::chrono::high_resolution_clock::now();
    motor.enable(MOTOR_ID);
    auto enableEnd = std::chrono::high_resolution_clock::now();
    std::cout<< "enable time: " << std::chrono::duration<double, std::milli>(enableEnd-enableStart).count() << " millis" << std::endl;
    std::cout<<"------ motor enabled ------"<< std::endl;
    
    motor.command(MOTOR_ID, 0, 0, 0, 0, 0);
    std::this_thread::sleep_for(chrono::microseconds(1000));
    resp_0 = motor.cycle_responses[1];
    std::cout<< "Homming response: " << resp_0 << std::endl;
    
    // Before connecting Arduino
    std::this_thread::sleep_for(chrono::seconds(3));

    try {
        // Connect to Arduino
        arduino.setPort("/dev/ttyACM0"); 
        arduino.setBaudrate(115200);
        serial::Timeout to = serial::Timeout::simpleTimeout(1000);
        arduino.setTimeout(to);
        arduino.open();

        arduino.setDTR(false);    // Lower DTR (off)
        std::this_thread::sleep_for(chrono::milliseconds(100));
        arduino.setDTR(true);     // Raise DTR (on)
        std::this_thread::sleep_for(chrono::seconds(2)); // Allow Arduino time to reboot

        if (!arduino.isOpen()) {
            cerr << "Failed to open Arduino port!" << endl;
            return -1;
        }

        std::cout << "------ Arduino connected ------" << endl;

    } catch (const std::exception& e) {
        cerr << "Exception: " << e.what() << endl;
    }
     
    // Before executing main loop
    std::this_thread::sleep_for(chrono::seconds(1));

    CartpoleSim2real model;
    model.run();

    motor.disable(MOTOR_ID);
    
    return 0;
}
