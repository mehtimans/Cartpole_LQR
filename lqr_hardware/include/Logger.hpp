#ifndef LOGGER_HPP
#define LOGGER_HPP


#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>


class Logger {
public:
    Logger(const std::string& log_dir, const std::string& flags);
    void logData(const std::vector<float>& data);


private:
    std::ofstream log_file_;
    void createLogFile(const std::string& log_dir, const std::string& flags);

};

#endif






