cmake_minimum_required(VERSION 3.16)
project(cyclo_teleoperation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(cyclo_motion_controller_core REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(robotis_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

add_library(cyclo_teleoperation_runtime SHARED
  src/core/controller_constraints.cpp
  src/core/joint_trajectory_interpolator.cpp
)
target_include_directories(cyclo_teleoperation_runtime PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(cyclo_teleoperation_runtime
  cyclo_motion_controller_core::cyclo_motion_controller_core
  Eigen3::Eigen
)
ament_target_dependencies(cyclo_teleoperation_runtime rclcpp)

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

add_library(cyclo_teleoperation_common_controller_plugins SHARED
  src/controllers/common/absolute_pose_mode.cpp
  src/controllers/common/movej_mode.cpp
  src/controllers/common/relative_pose_mode.cpp
)
target_include_directories(cyclo_teleoperation_common_controller_plugins PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(cyclo_teleoperation_common_controller_plugins
  cyclo_teleoperation_runtime
  cyclo_motion_controller_core::cyclo_motion_controller_core
  Eigen3::Eigen
)
ament_target_dependencies(cyclo_teleoperation_common_controller_plugins pluginlib rclcpp)

add_library(cyclo_teleoperation_ai_worker_robot_plugin SHARED
  src/robots/ai_worker/ai_worker_control_interface.cpp
  src/robots/ai_worker/ai_worker_teleoperation.cpp
)
target_include_directories(cyclo_teleoperation_ai_worker_robot_plugin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(cyclo_teleoperation_ai_worker_robot_plugin
  cyclo_motion_controller_core::cyclo_motion_controller_core
  Eigen3::Eigen
)
ament_target_dependencies(cyclo_teleoperation_ai_worker_robot_plugin
  geometry_msgs pluginlib rclcpp robotis_interfaces sensor_msgs trajectory_msgs)

add_executable(cyclo_teleoperation_node
  src/core/pose_sequence_manager.cpp
  src/core/teleoperation_node.cpp
  src/cyclo_teleoperation_main.cpp
)
target_include_directories(cyclo_teleoperation_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(cyclo_teleoperation_node
  cyclo_teleoperation_runtime
  cyclo_teleoperation_qp
  cyclo_motion_controller_core::cyclo_motion_controller_core
  Eigen3::Eigen
)
ament_target_dependencies(cyclo_teleoperation_node
  pluginlib rclcpp sensor_msgs trajectory_msgs)

add_executable(cyclo_action_controller_node
  src/core/action_controller_node.cpp
  src/cyclo_action_controller_main.cpp
)
target_include_directories(cyclo_action_controller_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(cyclo_action_controller_node
  cyclo_teleoperation_qp
  cyclo_teleoperation_runtime
  cyclo_motion_controller_core::cyclo_motion_controller_core
  Eigen3::Eigen
)
ament_target_dependencies(cyclo_action_controller_node
  geometry_msgs pluginlib rclcpp robotis_interfaces sensor_msgs trajectory_msgs)

pluginlib_export_plugin_description_file(
  cyclo_teleoperation cyclo_teleoperation_controller_plugins.xml)
pluginlib_export_plugin_description_file(
  cyclo_teleoperation cyclo_teleoperation_robot_plugins.xml)

install(TARGETS
  cyclo_teleoperation_qp
  cyclo_teleoperation_runtime
  cyclo_teleoperation_common_controller_plugins
  cyclo_teleoperation_ai_worker_robot_plugin
  cyclo_teleoperation_node
  cyclo_action_controller_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(FILES
  cyclo_teleoperation_controller_plugins.xml
  cyclo_teleoperation_robot_plugins.xml
  DESTINATION share/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_dependencies(cyclo_motion_controller_core Eigen3 pluginlib rclcpp)

ament_package()
