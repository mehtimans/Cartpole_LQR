
cmake_minimum_required(VERSION 3.10)
project(cartpole_sim2real)

# Set the C++ standard to C++11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(YAML_LIBRARY /usr/lib/x86_64-linux-gnu)

# Set path to LibTorch (CPU-only)
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")

# LibTorch
find_package(Torch REQUIRED)

# Eigen3 
find_package(Eigen3 REQUIRED)

# lcm
find_package(PkgConfig REQUIRED)
pkg_check_modules(LCM REQUIRED lcm)

# Threads
find_package(Threads REQUIRED)

include_directories(${LCM_INCLUDE_DIRS})
link_directories(${LCM_LIBRARY_DIRS})

include_directories("include/")
include_directories("types/")
include_directories("/usr/include/")
include_directories("/usr/local/include/eigen3/")
include_directories("script/")


# Include Serial
include_directories(${CMAKE_SOURCE_DIR}/serial/include)
link_directories(${CMAKE_SOURCE_DIR}/serial/build)

add_library(socket_can SHARED 
    "include/SocketCAN.h" "src/SocketCAN.cpp"
    "include/CANAdapter.h" "src/CANAdapter.cpp"
)
add_library(actuator SHARED 
    "include/actuator.hpp" "src/actuator.cpp"
)

add_library(logger SHARED 
    "include/Logger.hpp" "src/Logger.cpp"
)

add_library(CartpoleLQR SHARED 
    "script/CartpoleLQR.hpp" "script/CartpoleLQR.cpp"
)
target_include_directories(CartpoleLQR PUBLIC ${TORCH_INCLUDE_DIRS})

#### Create executable
add_executable(main_lqr "script/main.cpp")

#### Link Torch libraries
target_link_libraries(main_lqr PUBLIC "${TORCH_LIBRARIES}" actuator socket_can logger CartpoleLQR ${LCM_LIBRARIES} pthread serial Threads::Threads)

install(TARGETS 
        main_lqr
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib 
)


