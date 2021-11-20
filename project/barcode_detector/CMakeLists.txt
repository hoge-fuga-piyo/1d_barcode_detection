set(PROJ_NAME BarcodeDetector)
project(${PROJ_NAME})

# enable c++17
set(CMAKE_CXX_STANDARD 17)

# internal
file(GLOB PROJ_SOURCE *.cpp)
file(GLOB PROJ_INCLUDE *.h *.hpp)

# shared
include_directories(${PROJ_SHARED_DIR})

#target
add_executable(${PROJ_NAME}
  ${PROJ_SOURCE} ${PROJ_INCLUDE}
  ${PROJ_SHARED_SOURCE} ${PROJ_SHARED_INCLUDE}
)

target_link_libraries(${PROJ_NAME}
  ${OpenCV_LIBS}
)
