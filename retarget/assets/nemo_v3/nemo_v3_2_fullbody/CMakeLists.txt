cmake_minimum_required(VERSION 3.8)
project(nemo_v3_2_fullbody)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  config
  launch
  meshes
  urdf
  xacro
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
