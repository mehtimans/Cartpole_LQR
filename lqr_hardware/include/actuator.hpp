#ifndef ACTUATOR_H
#define ACTUATOR_H

#include <linux/can.h>
#include <string>

#include <lcm/lcm-cpp.hpp>

#include "CANFrame.h"
#include "SocketCAN.h"

#include "actuatorlcm/actuator_response_t.hpp"


class Actuator {
    private:
        std::string __can_conf_1;
        std::string __can_conf_2;
        std::string __can_conf_3;
        std::string __can_conf_4;
        SocketCAN* adapter;
        lcm::LCM lcm;
        actuatorlcm::actuator_response_t DATA;
    public:
        int m_index;
        float *cycle_responses;
        int responseCount;
        bool available;
        double timeout;
        int r;
        float *result;
        int s;
        int __data[8] = {0,0,0,0,0,0,0,0};
        float __pose_shift;
        struct can_frame frame;
        struct can_filter rfilter[1];
        Actuator(std::string can_index);
        void rx_handler(can_frame_t*);
        void enable(int id);
        void disable(int id);

        void zero(int id);
        void command(int id,double p_d,double v_d,double kp,double kd,double ff);
        float * __unpack(int motor_id);
        float * __pack2(double p_d,double v_d,double kp,double kd,double ff);
        void decToBinary(int n, int* binaryNum);
        int binaryToDec(int* binaryNum);
};

#endif
