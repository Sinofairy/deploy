cmake_minimum_required(VERSION 2.8)
set(CMAKE_C_COMPILER /opt/gcc-ubuntu-9.3.0-2020.03-x86_64-aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /opt/gcc-ubuntu-9.3.0-2020.03-x86_64-aarch64-linux-gnu/bin/aarch64-linux-gnu-g++)

project(J5_Sample)

option(RELEASE_LIB "build version of release" ON)

if (NOT (${RELEASE_LIB}))
  list(APPEND CMAKE_C_FLAGS " -g -Og")
endif()
list(APPEND CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++11 -pthread -Wall")

include_directories(
  # system image
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/hobot/include
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/hobotlog/include
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/protobuf/include
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/jsoncpp/include
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/zeroMQ/include
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/appsdk/appuser/include
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/dnn/include
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/image_utils/include
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/opencv/include
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/uWS/include
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/zlib/include
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/openssl/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/modules
  ${CMAKE_CURRENT_SOURCE_DIR}/include/message
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
)
link_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/hobot/lib
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/hobotlog/lib
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/protobuf/lib
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/jsoncpp/lib
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/dnn/lib
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/image_utils/lib
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/opencv/lib
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/uWS/lib
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/zlib/lib
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/zeroMQ/lib
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/openssl/lib
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/appsdk/appuser/lib
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/appsdk/appuser/lib/hbmedia
)
add_definitions(-DHR_POSIX)

file(GLOB PROTOCOL_FILE
  src/utils/protocol/*.cc)

set(SOURCE_FILES
        src/main.cc
        src/modules/inference_module.cc
        src/modules/web_display_module.cc
        src/modules/codec_module.cc
        src/uws_server/uws_server.cc
        src/modules/vio_module.cc
        src/modules/input/input_module.cc
        src/modules/input/camera_input_module.cc
        src/modules/input/network_input_module.cc
        src/modules/input/network_receiver.cc
        ${PROTOCOL_FILE}
        )


add_executable(J5_Sample ${SOURCE_FILES})

set(J5_MEDIA_LIBS vio vio_client vpm cam multimedia)
set(HOBOT_LIBS hobotsdk hobotthread hobotlog dnn image_utils)
set(BASE_LIBRARIES zmq protobuf.a opencv_world zlib uWS ssl crypto z dl m pthread rt)
target_link_libraries(J5_Sample ${HOBOT_LIBS} ${BASE_LIBRARIES} ${J5_MEDIA_LIBS})

set(OUTPUT_ROOT ${CMAKE_SOURCE_DIR}/output/)
set(MY_OUTPUT_ROOT ${OUTPUT_ROOT}/${PROJECT_NAME}/)
#install(DIRECTORY
#        ${CMAKE_CURRENT_SOURCE_DIR}/configs/
#        DESTINATION ${MY_OUTPUT_ROOT}/configs/)
install(TARGETS ${PROJECT_NAME}
        DESTINATION ${MY_OUTPUT_ROOT}/)
