cmake_minimum_required(VERSION 3.8)
project(motion_controller_models)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

set(LOCAL_MODELS_DIR "${CMAKE_CURRENT_LIST_DIR}/models")
set(FALLBACK_MODELS_DIR "${CMAKE_CURRENT_LIST_DIR}/../motion_controller_ros/models")

if(EXISTS "${LOCAL_MODELS_DIR}")
  install(DIRECTORY "${LOCAL_MODELS_DIR}/"
    DESTINATION "share/${PROJECT_NAME}/models"
  )
elseif(EXISTS "${FALLBACK_MODELS_DIR}")
  # Backward-compatible: until models are physically moved into this package,
  # install them from the legacy location.
  install(DIRECTORY "${FALLBACK_MODELS_DIR}/"
    DESTINATION "share/${PROJECT_NAME}/models"
  )
else()
  message(FATAL_ERROR
    "No models directory found. Expected one of:\n"
    "  - ${LOCAL_MODELS_DIR}\n"
    "  - ${FALLBACK_MODELS_DIR}"
  )
endif()

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/launch")
  install(DIRECTORY launch/
    DESTINATION "share/${PROJECT_NAME}/launch"
  )
endif()

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/rviz")
  install(DIRECTORY rviz/
    DESTINATION "share/${PROJECT_NAME}/rviz"
  )
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
