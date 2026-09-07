

#include "Logger.hpp"
#include <chrono>
#include <fstream>
#include <ctime>
#include <filesystem>
#include <vector>
#include <sstream>

Logger::Logger(const std::string& log_dir, const std::string& flags) {
    createLogFile(log_dir, flags);
}

void Logger::createLogFile(const std::string& log_dir, const std::string& flags) {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);

    std::stringstream folder_name_ss;
    folder_name_ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d_%H-%M-%S");
    std::string folder_name = folder_name_ss.str();

    std::filesystem::path full_dir = std::filesystem::path(log_dir) / folder_name;
    if (!std::filesystem::exists(full_dir)) {
        std::filesystem::create_directories(full_dir);
    }

    std::string log_file_path = (full_dir / "lqr_data.csv").string();
    log_file_.open(log_file_path);
    log_file_ << flags << "\n";
}

void Logger::logData(const std::vector<float>& data) {
    if (log_file_.is_open()) {
        // Write each float from the vector as a CSV line
        for (size_t i = 0; i < data.size(); ++i) {
            log_file_ << data[i]; 
            if (i != data.size() - 1) {
                log_file_ << ",";  
            }
        }

        log_file_ << "\n";  
        log_file_.flush();  
    } else {
        std::cerr << "Error: Log file is not open!" << std::endl;
    }
}