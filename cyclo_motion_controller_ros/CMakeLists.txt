cmake_minimum_required(VERSION 3.8)
project(cyclo_motion_controller_ros)

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
find_package(osqp_eigen_vendor REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(robotis_interfaces REQUIRED)

# Setup fallback paths first
set(CYCLO_MOTION_CONTROLLER_CORE_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../cyclo_motion_controller_core/include")
set(CYCLO_MOTION_CONTROLLER_CORE_NAMESPACE_INCLUDE_DIR "${CYCLO_MOTION_CONTROLLER_CORE_INCLUDE_DIR}/cyclo_motion_controller_core")
set(CYCLO_MOTION_CONTROLLER_CORE_LIB_DIR "${CMAKE_SOURCE_DIR}/../../install/cyclo_motion_controller_core/lib")

# Try to find cyclo_motion_controller_core
find_package(cyclo_motion_controller_core QUIET)

# Setup include paths and libraries
if(cyclo_motion_controller_core_FOUND)
  message(STATUS "Found cyclo_motion_controller_core via find_package")
else()
  message(STATUS "cyclo_motion_controller_core not found via find_package, using fallback paths")

  # Check if library exists
  if(EXISTS "${CYCLO_MOTION_CONTROLLER_CORE_LIB_DIR}/libcyclo_motion_controller_core.so")
    set(CYCLO_MOTION_CONTROLLER_CORE_LIB "${CYCLO_MOTION_CONTROLLER_CORE_LIB_DIR}/libcyclo_motion_controller_core.so")
    message(STATUS "Found cyclo_motion_controller_core library at: ${CYCLO_MOTION_CONTROLLER_CORE_LIB}")

    # Create imported target
    if(NOT TARGET cyclo_motion_controller_core)
      add_library(cyclo_motion_controller_core SHARED IMPORTED GLOBAL)
      set_target_properties(cyclo_motion_controller_core PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CYCLO_MOTION_CONTROLLER_CORE_INCLUDE_DIR}"
        IMPORTED_LOCATION "${CYCLO_MOTION_CONTROLLER_CORE_LIB}"
      )
    endif()
  else()
    message(FATAL_ERROR "Could not find libcyclo_motion_controller_core.so in ${CYCLO_MOTION_CONTROLLER_CORE_LIB_DIR}")
  endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIR})

# Create executable
add_executable(vr_controller_node
  src/nodes/ai_worker/vr_controller_node.cpp
)

add_executable(ai_worker_movej_controller_node
  src/nodes/ai_worker/ai_worker_movej_controller_node.cpp
)

add_executable(ai_worker_movel_controller_node
  src/nodes/ai_worker/ai_worker_movel_controller_node.cpp
)

add_executable(leader_controller_node
  src/nodes/ai_worker/leader_controller_node.cpp
)

add_executable(wholebody_controller_node
  src/nodes/ai_worker/wholebody_controller_node.cpp
)

add_executable(omx_movej_controller_node
  src/nodes/omx/omx_movej_controller_node.cpp
)

add_executable(omx_movel_controller_node
  src/nodes/omx/omx_movel_controller_node.cpp
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

add_executable(leader_trajectory_sequence_node
  src/utils/leader_trajectory_sequence_node.cpp
)

add_executable(movel_showcase_demo_node
  src/utils/movel_showcase_demo_node.cpp
)

add_executable(orientation_reference_demo_node
  src/utils/orientation_reference_demo_node.cpp
)

add_executable(vr_redundancy_reference_node
  src/utils/vr_redundancy_reference_node.cpp
)

add_executable(wholebody_reference_node
  src/utils/wholebody_reference_node.cpp
)

# Add include directories to the target
message(STATUS "CYCLO_MOTION_CONTROLLER_CORE_INCLUDE_DIR = ${CYCLO_MOTION_CONTROLLER_CORE_INCLUDE_DIR}")

target_include_directories(vr_controller_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${CYCLO_MOTION_CONTROLLER_CORE_NAMESPACE_INCLUDE_DIR}
)

target_include_directories(ai_worker_movej_controller_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${CYCLO_MOTION_CONTROLLER_CORE_NAMESPACE_INCLUDE_DIR}
)

target_include_directories(ai_worker_movel_controller_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${CYCLO_MOTION_CONTROLLER_CORE_NAMESPACE_INCLUDE_DIR}
)

target_include_directories(leader_controller_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${CYCLO_MOTION_CONTROLLER_CORE_NAMESPACE_INCLUDE_DIR}
)

target_include_directories(wholebody_controller_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${CYCLO_MOTION_CONTROLLER_CORE_NAMESPACE_INCLUDE_DIR}
)

target_include_directories(omx_movej_controller_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${CYCLO_MOTION_CONTROLLER_CORE_NAMESPACE_INCLUDE_DIR}
)

target_include_directories(omx_movel_controller_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${CYCLO_MOTION_CONTROLLER_CORE_NAMESPACE_INCLUDE_DIR}
)

target_include_directories(omy_movej_controller_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${CYCLO_MOTION_CONTROLLER_CORE_NAMESPACE_INCLUDE_DIR}
)

target_include_directories(omy_movel_controller_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${CYCLO_MOTION_CONTROLLER_CORE_NAMESPACE_INCLUDE_DIR}
)

# Link libraries
target_include_directories(vr_controller_node PRIVATE
  ${rclcpp_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${std_srvs_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${trajectory_msgs_INCLUDE_DIRS}
  ${ament_index_cpp_INCLUDE_DIRS}
)
target_link_libraries(vr_controller_node PRIVATE
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${std_srvs_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${trajectory_msgs_LIBRARIES}
  ${ament_index_cpp_LIBRARIES}
  ${rclcpp_TARGETS}
  ${std_msgs_TARGETS}
  ${std_srvs_TARGETS}
  ${geometry_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${trajectory_msgs_TARGETS}
  ${ament_index_cpp_TARGETS}
)

target_include_directories(ai_worker_movej_controller_node PRIVATE
  ${rclcpp_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${trajectory_msgs_INCLUDE_DIRS}
  ${ament_index_cpp_INCLUDE_DIRS}
)
target_link_libraries(ai_worker_movej_controller_node PRIVATE
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${trajectory_msgs_LIBRARIES}
  ${ament_index_cpp_LIBRARIES}
  ${rclcpp_TARGETS}
  ${std_msgs_TARGETS}
  ${geometry_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${trajectory_msgs_TARGETS}
  ${ament_index_cpp_TARGETS}
)

target_include_directories(ai_worker_movel_controller_node PRIVATE
  ${robotis_interfaces_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${trajectory_msgs_INCLUDE_DIRS}
)
target_link_libraries(ai_worker_movel_controller_node PRIVATE
  ${robotis_interfaces_LIBRARIES}
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${trajectory_msgs_LIBRARIES}
  ${robotis_interfaces_TARGETS}
  ${rclcpp_TARGETS}
  ${std_msgs_TARGETS}
  ${geometry_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${trajectory_msgs_TARGETS}
)

target_include_directories(leader_controller_node PRIVATE
  ${rclcpp_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${std_srvs_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${trajectory_msgs_INCLUDE_DIRS}
  ${ament_index_cpp_INCLUDE_DIRS}
)
target_link_libraries(leader_controller_node PRIVATE
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${std_srvs_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${trajectory_msgs_LIBRARIES}
  ${ament_index_cpp_LIBRARIES}
  ${rclcpp_TARGETS}
  ${std_msgs_TARGETS}
  ${std_srvs_TARGETS}
  ${geometry_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${trajectory_msgs_TARGETS}
  ${ament_index_cpp_TARGETS}
)

target_include_directories(omx_movej_controller_node PRIVATE
  ${rclcpp_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${trajectory_msgs_INCLUDE_DIRS}
)
target_link_libraries(omx_movej_controller_node PRIVATE
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${trajectory_msgs_LIBRARIES}
  ${rclcpp_TARGETS}
  ${std_msgs_TARGETS}
  ${geometry_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${trajectory_msgs_TARGETS}
)

target_include_directories(omx_movel_controller_node PRIVATE
  ${robotis_interfaces_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${trajectory_msgs_INCLUDE_DIRS}
)
target_link_libraries(omx_movel_controller_node PRIVATE
  ${robotis_interfaces_LIBRARIES}
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${trajectory_msgs_LIBRARIES}
  ${robotis_interfaces_TARGETS}
  ${rclcpp_TARGETS}
  ${std_msgs_TARGETS}
  ${geometry_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${trajectory_msgs_TARGETS}
)

target_include_directories(omy_movej_controller_node PRIVATE
  ${rclcpp_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${trajectory_msgs_INCLUDE_DIRS}
)
target_link_libraries(omy_movej_controller_node PRIVATE
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${trajectory_msgs_LIBRARIES}
  ${rclcpp_TARGETS}
  ${std_msgs_TARGETS}
  ${geometry_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${trajectory_msgs_TARGETS}
)

target_include_directories(omy_movel_controller_node PRIVATE
  ${robotis_interfaces_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${trajectory_msgs_INCLUDE_DIRS}
)
target_link_libraries(omy_movel_controller_node PRIVATE
  ${robotis_interfaces_LIBRARIES}
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${trajectory_msgs_LIBRARIES}
  ${robotis_interfaces_TARGETS}
  ${rclcpp_TARGETS}
  ${std_msgs_TARGETS}
  ${geometry_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${trajectory_msgs_TARGETS}
)

target_include_directories(interactive_marker_node PRIVATE
  ${robotis_interfaces_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${visualization_msgs_INCLUDE_DIRS}
  ${interactive_markers_INCLUDE_DIRS}
  ${tf2_ros_INCLUDE_DIRS}
)
target_link_libraries(interactive_marker_node PRIVATE
  ${robotis_interfaces_LIBRARIES}
  ${rclcpp_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${visualization_msgs_LIBRARIES}
  ${interactive_markers_LIBRARIES}
  ${tf2_ros_LIBRARIES}
  ${robotis_interfaces_TARGETS}
  ${rclcpp_TARGETS}
  ${geometry_msgs_TARGETS}
  ${visualization_msgs_TARGETS}
  ${interactive_markers_TARGETS}
  ${tf2_ros_TARGETS}
)

target_include_directories(reference_checker_node PRIVATE
  ${rclcpp_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
)
target_include_directories(vr_redundancy_reference_node PRIVATE
  ${rclcpp_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${tf2_ros_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CYCLO_MOTION_CONTROLLER_CORE_NAMESPACE_INCLUDE_DIR}
)
target_include_directories(wholebody_reference_node PRIVATE
  ${rclcpp_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${tf2_ros_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)
target_include_directories(wholebody_controller_node PRIVATE
  ${rclcpp_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${std_srvs_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${trajectory_msgs_INCLUDE_DIRS}
)
target_link_libraries(reference_checker_node PRIVATE
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${rclcpp_TARGETS}
  ${std_msgs_TARGETS}
  ${geometry_msgs_TARGETS}
)
target_link_libraries(vr_redundancy_reference_node PRIVATE
  ${rclcpp_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${tf2_ros_LIBRARIES}
  ${rclcpp_TARGETS}
  ${geometry_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${tf2_ros_TARGETS}
)
target_link_libraries(wholebody_reference_node PRIVATE
  ${rclcpp_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${tf2_ros_LIBRARIES}
  ${rclcpp_TARGETS}
  ${geometry_msgs_TARGETS}
  ${tf2_ros_TARGETS}
)
target_link_libraries(wholebody_controller_node PRIVATE
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${std_srvs_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${trajectory_msgs_LIBRARIES}
  ${rclcpp_TARGETS}
  ${std_msgs_TARGETS}
  ${std_srvs_TARGETS}
  ${geometry_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${trajectory_msgs_TARGETS}
)

target_include_directories(leader_trajectory_sequence_node PRIVATE
  ${rclcpp_INCLUDE_DIRS}
  ${trajectory_msgs_INCLUDE_DIRS}
)
target_link_libraries(leader_trajectory_sequence_node PRIVATE
  ${rclcpp_LIBRARIES}
  ${trajectory_msgs_LIBRARIES}
  ${rclcpp_TARGETS}
  ${trajectory_msgs_TARGETS}
)

target_include_directories(movel_showcase_demo_node PRIVATE
  ${robotis_interfaces_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${trajectory_msgs_INCLUDE_DIRS}
  ${visualization_msgs_INCLUDE_DIRS}
  ${tf2_ros_INCLUDE_DIRS}
  ${ament_index_cpp_INCLUDE_DIRS}
  ${CYCLO_MOTION_CONTROLLER_CORE_NAMESPACE_INCLUDE_DIR}
)
target_link_libraries(movel_showcase_demo_node PRIVATE
  ${robotis_interfaces_LIBRARIES}
  ${rclcpp_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${trajectory_msgs_LIBRARIES}
  ${visualization_msgs_LIBRARIES}
  ${tf2_ros_LIBRARIES}
  ${ament_index_cpp_LIBRARIES}
  ${robotis_interfaces_TARGETS}
  ${rclcpp_TARGETS}
  ${geometry_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${trajectory_msgs_TARGETS}
  ${visualization_msgs_TARGETS}
  ${tf2_ros_TARGETS}
  ${ament_index_cpp_TARGETS}
)

target_include_directories(orientation_reference_demo_node PRIVATE
  ${robotis_interfaces_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${tf2_ros_INCLUDE_DIRS}
)
target_link_libraries(orientation_reference_demo_node PRIVATE
  ${robotis_interfaces_LIBRARIES}
  ${rclcpp_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${tf2_ros_LIBRARIES}
  ${robotis_interfaces_TARGETS}
  ${rclcpp_TARGETS}
  ${geometry_msgs_TARGETS}
  ${tf2_ros_TARGETS}
)

target_link_libraries(vr_controller_node PRIVATE
  Eigen3::Eigen
  pinocchio::pinocchio
  OsqpEigen::OsqpEigen
)

target_link_libraries(ai_worker_movej_controller_node PRIVATE
  Eigen3::Eigen
  pinocchio::pinocchio
  OsqpEigen::OsqpEigen
)

target_link_libraries(ai_worker_movel_controller_node PRIVATE
  Eigen3::Eigen
  pinocchio::pinocchio
  OsqpEigen::OsqpEigen
)

target_link_libraries(leader_controller_node PRIVATE
  Eigen3::Eigen
  pinocchio::pinocchio
  OsqpEigen::OsqpEigen
)

target_link_libraries(omx_movej_controller_node PRIVATE
  Eigen3::Eigen
  pinocchio::pinocchio
  OsqpEigen::OsqpEigen
)

target_link_libraries(omx_movel_controller_node PRIVATE
  Eigen3::Eigen
  pinocchio::pinocchio
  OsqpEigen::OsqpEigen
)

target_link_libraries(omy_movej_controller_node PRIVATE
  Eigen3::Eigen
  pinocchio::pinocchio
  OsqpEigen::OsqpEigen
)

target_link_libraries(omy_movel_controller_node PRIVATE
  Eigen3::Eigen
  pinocchio::pinocchio
  OsqpEigen::OsqpEigen
)

target_link_libraries(vr_redundancy_reference_node PRIVATE
  Eigen3::Eigen
  pinocchio::pinocchio
  OsqpEigen::OsqpEigen
)
target_link_libraries(wholebody_controller_node PRIVATE
  Eigen3::Eigen
  pinocchio::pinocchio
  OsqpEigen::OsqpEigen
)
target_link_libraries(wholebody_reference_node PRIVATE
  Eigen3::Eigen
)

# Link cyclo_motion_controller_core
if(cyclo_motion_controller_core_FOUND)
  target_link_libraries(vr_controller_node PRIVATE cyclo_motion_controller_core::cyclo_motion_controller_core)
  target_link_libraries(ai_worker_movej_controller_node PRIVATE cyclo_motion_controller_core::cyclo_motion_controller_core)
  target_link_libraries(ai_worker_movel_controller_node PRIVATE cyclo_motion_controller_core::cyclo_motion_controller_core)
  target_link_libraries(leader_controller_node PRIVATE cyclo_motion_controller_core::cyclo_motion_controller_core)
  target_link_libraries(wholebody_controller_node PRIVATE cyclo_motion_controller_core::cyclo_motion_controller_core)
  target_link_libraries(omx_movej_controller_node PRIVATE cyclo_motion_controller_core::cyclo_motion_controller_core)
  target_link_libraries(omx_movel_controller_node PRIVATE cyclo_motion_controller_core::cyclo_motion_controller_core)
  target_link_libraries(omy_movej_controller_node PRIVATE cyclo_motion_controller_core::cyclo_motion_controller_core)
  target_link_libraries(omy_movel_controller_node PRIVATE cyclo_motion_controller_core::cyclo_motion_controller_core)
  target_link_libraries(movel_showcase_demo_node PRIVATE cyclo_motion_controller_core::cyclo_motion_controller_core)
  target_link_libraries(vr_redundancy_reference_node PRIVATE cyclo_motion_controller_core::cyclo_motion_controller_core)
else()
  # Link with full path for fallback
  target_link_libraries(vr_controller_node PRIVATE ${CYCLO_MOTION_CONTROLLER_CORE_LIB})
  target_link_libraries(ai_worker_movej_controller_node PRIVATE ${CYCLO_MOTION_CONTROLLER_CORE_LIB})
  target_link_libraries(ai_worker_movel_controller_node PRIVATE ${CYCLO_MOTION_CONTROLLER_CORE_LIB})
  target_link_libraries(leader_controller_node PRIVATE ${CYCLO_MOTION_CONTROLLER_CORE_LIB})
  target_link_libraries(wholebody_controller_node PRIVATE ${CYCLO_MOTION_CONTROLLER_CORE_LIB})
  target_link_libraries(omx_movej_controller_node PRIVATE ${CYCLO_MOTION_CONTROLLER_CORE_LIB})
  target_link_libraries(omx_movel_controller_node PRIVATE ${CYCLO_MOTION_CONTROLLER_CORE_LIB})
  target_link_libraries(omy_movej_controller_node PRIVATE ${CYCLO_MOTION_CONTROLLER_CORE_LIB})
  target_link_libraries(omy_movel_controller_node PRIVATE ${CYCLO_MOTION_CONTROLLER_CORE_LIB})
  target_link_libraries(movel_showcase_demo_node PRIVATE ${CYCLO_MOTION_CONTROLLER_CORE_LIB})
  target_link_libraries(vr_redundancy_reference_node PRIVATE ${CYCLO_MOTION_CONTROLLER_CORE_LIB})
endif()

# Install executable
install(TARGETS vr_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS ai_worker_movej_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS ai_worker_movel_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS leader_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS wholebody_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS omx_movej_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS omx_movel_controller_node
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

install(TARGETS leader_trajectory_sequence_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS movel_showcase_demo_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS orientation_reference_demo_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS vr_redundancy_reference_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS wholebody_reference_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
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
