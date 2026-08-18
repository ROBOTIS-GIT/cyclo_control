cmake_minimum_required(VERSION 3.16)
project(cyclo_teleoperation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(cyclo_motion_controller_core REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(robotis_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

add_library(cyclo_teleoperation_qp SHARED
  src/core/soft_hold.cpp
  src/core/teleoperation_qp.cpp
)
target_include_directories(cyclo_teleoperation_qp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(cyclo_teleoperation_qp PUBLIC
  cyclo_motion_controller_core::cyclo_motion_controller_core
  Eigen3::Eigen
)

add_library(cyclo_teleoperation_modes SHARED
  src/robots/ai_worker/movej_mode.cpp
  src/robots/ai_worker/pose_sequence_manager.cpp
  src/robots/ai_worker/elbow_up_leader_mode.cpp
)
target_include_directories(cyclo_teleoperation_modes PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(cyclo_teleoperation_modes
  cyclo_motion_controller_core::cyclo_motion_controller_core
  Eigen3::Eigen
)
ament_target_dependencies(cyclo_teleoperation_modes pluginlib rclcpp)

add_executable(cyclo_teleoperation_node
  src/cyclo_teleoperation_node.cpp
  src/robots/ai_worker/ai_worker_profile.cpp
)
target_include_directories(cyclo_teleoperation_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(cyclo_teleoperation_node
  cyclo_teleoperation_qp
  cyclo_teleoperation_modes
  cyclo_motion_controller_core::cyclo_motion_controller_core
  Eigen3::Eigen
)
ament_target_dependencies(cyclo_teleoperation_node
  ament_index_cpp pluginlib rclcpp robotis_interfaces sensor_msgs trajectory_msgs)

pluginlib_export_plugin_description_file(
  cyclo_teleoperation cyclo_teleoperation_plugins.xml)

install(TARGETS
  cyclo_teleoperation_qp
  cyclo_teleoperation_modes
  cyclo_teleoperation_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(FILES cyclo_teleoperation_plugins.xml
  DESTINATION share/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_dependencies(cyclo_motion_controller_core Eigen3 pluginlib rclcpp)

ament_package()
