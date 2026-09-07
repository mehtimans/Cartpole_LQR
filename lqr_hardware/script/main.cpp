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
#include "CartpoleLQR.hpp"

#define MOTOR_ID 0x07        // 0x07

// Global 
serial::Serial arduino;            
Actuator motor("can0"); 
float resp_0 = 0;

int main() {
    // log data 
    std::filesystem::path current_path = std::filesystem::current_path();  
    std::string log_dir = (current_path.parent_path() / "logs").string();
    std::string flags = "time_sec,x_cart,vel_cart,ang_pole,vel_pole,torque_input,x_cart_des,"
                        "iteration_duration_s,bytesWaiting,current_resp_act,"
                        "current_resp_loop,error_position_pd,error_position_angle_pd";
    
    Logger logger(log_dir, flags);

    auto enableStart = std::chrono::high_resolution_clock::now();
    motor.enable(MOTOR_ID);
    auto enableEnd = std::chrono::high_resolution_clock::now();
    std::cout<< "enable time: " << std::chrono::duration<double, std::milli>(enableEnd-enableStart).count() << " millis" << std::endl;
    std::cout<<"------ motor enabled ------"<< std::endl;
    
    motor.command(MOTOR_ID, 0, 0, 0, 0, 0);
    std::this_thread::sleep_for(std::chrono::microseconds(1000));
    resp_0 = motor.cycle_responses[1];
    std::cout<< "Homing response: " << resp_0 << std::endl;
    
    // Before connecting Arduino
    std::this_thread::sleep_for(std::chrono::seconds(3));

    try {
        // Connect to Arduino
        arduino.setPort("/dev/ttyACM0"); 
        arduino.setBaudrate(115200);
        serial::Timeout to = serial::Timeout::simpleTimeout(1000);
        arduino.setTimeout(to);
        arduino.open();

        arduino.setDTR(false);    // Lower DTR (off)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        arduino.setDTR(true);     // Raise DTR (on)
        std::this_thread::sleep_for(std::chrono::seconds(2)); // Allow Arduino time to reboot

        if (!arduino.isOpen()) {
            std::cerr << "Failed to open Arduino port!" << std::endl;
            return -1;
        }

        std::cout << "------ Arduino connected ------" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
     
    // Before executing main loop
    std::this_thread::sleep_for(std::chrono::seconds(1));

    CartpoleLQR model(logger);
    model.run();

    motor.disable(MOTOR_ID);
    
    return 0;
}
