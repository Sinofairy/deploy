#!/bin/bash

SCRIPTS_DIR=$(cd `dirname $0`; pwd)
ALL_PROJECT_DIR=$PWD
SOLUTION_ZOO_DIR=${ALL_PROJECT_DIR}/..

function build_clean() {
  cd ${SOLUTION_ZOO_DIR}
  rm -rf common/wrapper/*/output
  rm -rf common/wrapper/*/*/output
  rm -rf common/xstream_methods/*/output
  rm -rf common/xproto_plugins/*/output
  cd ${ALL_PROJECT_DIR}
  rm build/ output/ deploy/ -rf
}

if [ $# -eq 1 ];then
  HOBOT_COMPILE_MODE=${1}
  if [ x"${HOBOT_COMPILE_MODE}" == x"clean" ];then
    build_clean
    exit
  else
    echo "error!!! compile cmd: ${HOBOT_COMPILE_MODE} is not supported"
  fi
fi

function go_build_all(){
  rm build -rf
  rm output -rf
  rm deploy -rf
  mkdir build
  cd build
  cmake .. $*
  echo "##############################################"
  echo $1
  make -j
  if [ $? -ne 0 ] ; then
    echo "failed to build"
    exit 1
   fi
  make install
  cd -
}

function prepare_depend_so(){
    mkdir -p deploy/lib

    # cp ./deps/aarch64/protobuf/lib/libprotobuf.so.10 ./deploy/lib -rf
    cp ./deps/aarch64/uWS/lib/libuWS.so ./deploy/lib -rf
    cp ./deps/aarch64/zeroMQ/lib/libzmq.so.5 ./deploy/lib -rf
    cp ./deps/aarch64/opencv/lib/libopencv_world.so.3.4 ./deploy/lib -rf
    cp ./deps/aarch64/dnn/lib/lib*.so ./deploy/lib -rf
    cp ./deps/aarch64/image_utils/lib/libimage_utils.so ./deploy/lib -rf
    cp ./output/* ./deploy -rf
}

function cp_configs(){
    mkdir -p deploy/configs
    cp configs/* ./deploy/configs/ -R
    cp run_sample.sh ./deploy -rf
    cp ./tools/webservice ./deploy -rf
    cp ./tools/webservice/start_nginx.sh ./deploy -rf
    chmod +x ./deploy/start_nginx.sh
}

go_build_all -DRELEASE_LIB=ON
prepare_depend_so
cp_configs
