cmake_minimum_required(VERSION 3.8)
project(franka_custom_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Franka REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(pid_controller_node src/PID_controller_node.cpp src/examples_common.cpp)
ament_target_dependencies(pid_controller_node rclcpp std_msgs)
target_link_libraries(pid_controller_node Franka::Franka)
target_link_libraries(pid_controller_node Eigen3::Eigen)
target_include_directories(pid_controller_node PRIVATE include)
install(TARGETS pid_controller_node DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
        DESTINATION include/
)

ament_package()
