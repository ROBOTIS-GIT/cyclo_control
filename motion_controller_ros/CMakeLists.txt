cmake_minimum_required(VERSION 3.8)
project(motion_controller_ros)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(interactive_markers REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(OsqpEigen REQUIRED)
find_package(ament_index_cpp REQUIRED)

# Setup fallback paths first
set(MOTION_CONTROLLER_CORE_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../motion_controller_core/include")
set(MOTION_CONTROLLER_CORE_LIB_DIR "${CMAKE_SOURCE_DIR}/../../install/motion_controller_core/lib")

# Try to find motion_controller_core
find_package(motion_controller_core QUIET)

# Setup include paths and libraries
if(motion_controller_core_FOUND)
  message(STATUS "Found motion_controller_core via find_package")
else()
  message(STATUS "motion_controller_core not found via find_package, using fallback paths")
  
  # Check if library exists
  if(EXISTS "${MOTION_CONTROLLER_CORE_LIB_DIR}/libmotion_controller_core.so")
    set(MOTION_CONTROLLER_CORE_LIB "${MOTION_CONTROLLER_CORE_LIB_DIR}/libmotion_controller_core.so")
    message(STATUS "Found motion_controller_core library at: ${MOTION_CONTROLLER_CORE_LIB}")
    
    # Create imported target
    if(NOT TARGET motion_controller_core)
      add_library(motion_controller_core SHARED IMPORTED GLOBAL)
      set_target_properties(motion_controller_core PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${MOTION_CONTROLLER_CORE_INCLUDE_DIR}"
        IMPORTED_LOCATION "${MOTION_CONTROLLER_CORE_LIB}"
      )
    endif()
  else()
    message(FATAL_ERROR "Could not find libmotion_controller_core.so in ${MOTION_CONTROLLER_CORE_LIB_DIR}")
  endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIR})

# Create executable
add_executable(ai_worker_controller_node 
  src/nodes/ai_worker/ai_worker_controller_node.cpp
)

add_executable(joint_space_controller_node 
  src/nodes/ai_worker/joint_space_controller_node.cpp
)

add_executable(leader_controller_node
  src/nodes/ai_worker/leader_controller_node.cpp
)

add_executable(omx_controller_node
  src/nodes/omx/omx_controller_node.cpp
)

add_executable(omx_movej_controller_node
  src/nodes/omx/omx_movej_controller_node.cpp
)

add_executable(omx_movel_controller_node
  src/nodes/omx/omx_movel_controller_node.cpp
)

add_executable(omy_controller_node
  src/nodes/omy/omy_controller_node.cpp
)

add_executable(omy_movej_controller_node
  src/nodes/omy/omy_movej_controller_node.cpp
)

add_executable(omy_movel_controller_node
  src/nodes/omy/omy_movel_controller_node.cpp
)

add_executable(interactive_marker_node
  src/utils/eef_interactive_marker_node.cpp
)

add_executable(reference_checker_node
  src/utils/reference_checker_node.cpp
)

# Add include directories to the target
message(STATUS "MOTION_CONTROLLER_CORE_INCLUDE_DIR = ${MOTION_CONTROLLER_CORE_INCLUDE_DIR}")

target_include_directories(ai_worker_controller_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${MOTION_CONTROLLER_CORE_INCLUDE_DIR}
)

target_include_directories(joint_space_controller_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${MOTION_CONTROLLER_CORE_INCLUDE_DIR}
)

target_include_directories(leader_controller_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${MOTION_CONTROLLER_CORE_INCLUDE_DIR}
)

target_include_directories(omx_controller_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${MOTION_CONTROLLER_CORE_INCLUDE_DIR}
)

target_include_directories(omx_movej_controller_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${MOTION_CONTROLLER_CORE_INCLUDE_DIR}
)

target_include_directories(omx_movel_controller_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${MOTION_CONTROLLER_CORE_INCLUDE_DIR}
)

target_include_directories(omy_controller_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${MOTION_CONTROLLER_CORE_INCLUDE_DIR}
)

target_include_directories(omy_movej_controller_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${MOTION_CONTROLLER_CORE_INCLUDE_DIR}
)

target_include_directories(omy_movel_controller_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${MOTION_CONTROLLER_CORE_INCLUDE_DIR}
)

# Link libraries
ament_target_dependencies(ai_worker_controller_node
  rclcpp
  std_msgs
  std_srvs
  geometry_msgs
  sensor_msgs
  trajectory_msgs
  ament_index_cpp
)

ament_target_dependencies(joint_space_controller_node
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  trajectory_msgs
  ament_index_cpp
)

ament_target_dependencies(leader_controller_node
  rclcpp
  std_msgs
  std_srvs
  geometry_msgs
  sensor_msgs
  trajectory_msgs
  ament_index_cpp
)

ament_target_dependencies(omx_controller_node
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  trajectory_msgs
)

ament_target_dependencies(omx_movej_controller_node
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  trajectory_msgs
)

ament_target_dependencies(omx_movel_controller_node
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  trajectory_msgs
)

ament_target_dependencies(omy_controller_node
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  trajectory_msgs
)

ament_target_dependencies(omy_movej_controller_node
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  trajectory_msgs
)

ament_target_dependencies(omy_movel_controller_node
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  trajectory_msgs
)

ament_target_dependencies(interactive_marker_node
  rclcpp
  geometry_msgs
  visualization_msgs
  interactive_markers
  tf2_ros
)

ament_target_dependencies(reference_checker_node
  rclcpp
  std_msgs
  geometry_msgs
)

target_link_libraries(ai_worker_controller_node
  Eigen3::Eigen
  pinocchio::pinocchio
  OsqpEigen::OsqpEigen
)

target_link_libraries(joint_space_controller_node
  Eigen3::Eigen
  pinocchio::pinocchio
  OsqpEigen::OsqpEigen
)

target_link_libraries(leader_controller_node
  Eigen3::Eigen
  pinocchio::pinocchio
  OsqpEigen::OsqpEigen
)

target_link_libraries(omx_controller_node
  Eigen3::Eigen
  pinocchio::pinocchio
  OsqpEigen::OsqpEigen
)

target_link_libraries(omx_movej_controller_node
  Eigen3::Eigen
  pinocchio::pinocchio
  OsqpEigen::OsqpEigen
)

target_link_libraries(omx_movel_controller_node
  Eigen3::Eigen
  pinocchio::pinocchio
  OsqpEigen::OsqpEigen
)

target_link_libraries(omy_controller_node
  Eigen3::Eigen
  pinocchio::pinocchio
  OsqpEigen::OsqpEigen
)

target_link_libraries(omy_movej_controller_node
  Eigen3::Eigen
  pinocchio::pinocchio
  OsqpEigen::OsqpEigen
)

target_link_libraries(omy_movel_controller_node
  Eigen3::Eigen
  pinocchio::pinocchio
  OsqpEigen::OsqpEigen
)

# Link motion_controller_core
if(motion_controller_core_FOUND)
  target_link_libraries(ai_worker_controller_node motion_controller_core::motion_controller_core)
  target_link_libraries(joint_space_controller_node motion_controller_core::motion_controller_core)
  target_link_libraries(leader_controller_node motion_controller_core::motion_controller_core)
  target_link_libraries(omx_controller_node motion_controller_core::motion_controller_core)
  target_link_libraries(omx_movej_controller_node motion_controller_core::motion_controller_core)
  target_link_libraries(omx_movel_controller_node motion_controller_core::motion_controller_core)
  target_link_libraries(omy_controller_node motion_controller_core::motion_controller_core)
  target_link_libraries(omy_movej_controller_node motion_controller_core::motion_controller_core)
  target_link_libraries(omy_movel_controller_node motion_controller_core::motion_controller_core)
else()
  # Link with full path for fallback
  target_link_libraries(ai_worker_controller_node ${MOTION_CONTROLLER_CORE_LIB})
  target_link_libraries(joint_space_controller_node ${MOTION_CONTROLLER_CORE_LIB})
  target_link_libraries(leader_controller_node ${MOTION_CONTROLLER_CORE_LIB})
  target_link_libraries(omx_controller_node ${MOTION_CONTROLLER_CORE_LIB})
  target_link_libraries(omx_movej_controller_node ${MOTION_CONTROLLER_CORE_LIB})
  target_link_libraries(omx_movel_controller_node ${MOTION_CONTROLLER_CORE_LIB})
  target_link_libraries(omy_controller_node ${MOTION_CONTROLLER_CORE_LIB})
  target_link_libraries(omy_movej_controller_node ${MOTION_CONTROLLER_CORE_LIB})
  target_link_libraries(omy_movel_controller_node ${MOTION_CONTROLLER_CORE_LIB})
endif()

# Install executable
install(TARGETS ai_worker_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS joint_space_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS leader_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS omx_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS omx_movej_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS omx_movel_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS omy_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS omy_movej_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS omy_movel_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS interactive_marker_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS reference_checker_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
