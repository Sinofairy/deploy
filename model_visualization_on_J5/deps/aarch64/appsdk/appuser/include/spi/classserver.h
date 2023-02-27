//
// Copyright 2020 Horizon
//
#ifndef CLASSSERVER_H
#define CLASSSERVER_H
#include <dirent.h>
#include <fcntl.h>
#include <getopt.h>
#include <linux/spi/spidev.h>
#include <linux/types.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include "zmq.h"

class Pub {
 public:
  Pub(std::string filepath) {
    if (filepath == "") {
      std::cout << "no zmq ipc file input" << std::endl;
    }
    file_path = filepath;
    if ((context = zmq_ctx_new()) == NULL) {
      std::cout << "pub context init failed" << std::endl;
    }
    InitPub();
  }
  ~Pub() {
    zmq_close(puber);
    zmq_ctx_destroy(context);
  }
  int SetSchedulePolicyPriority(int policy, int prioority) {
    if (context) {
      zmq_ctx_set(context, ZMQ_THREAD_SCHED_POLICY, policy);
      zmq_ctx_set(context, ZMQ_THREAD_PRIORITY, prioority);
    } else {
      return -1;
    }
    return 0;
  }

  int InitPub() {
    std::string addr = "";
    if (file_path.find("tcp") == std::string::npos) {
      if (access(file_path.c_str(), F_OK) == -1) {
        if (creat(file_path.c_str(), 0755) < 0) {
          std::cout << "create file failed" << std::endl;
          return -1;
        }
      }
      addr = "ipc://" + file_path;
    } else {
      addr = file_path;
    }

    if ((puber = zmq_socket(context, ZMQ_PUB)) == NULL) {
      std::cout << " pub socket init failed" << std::endl;
      return -1;
    }
    int hwm = 100;
    int rc = zmq_setsockopt(puber, ZMQ_SNDHWM, &hwm, sizeof(hwm));
    if (rc < 0) {
      std::cout << "set sndhwm failed" << std::endl;
      return -1;
    }

    int linger = 1000;
    rc = zmq_setsockopt(puber, ZMQ_LINGER, &linger, sizeof(linger));
    if (rc < 0) {
      std::cout << "set linger failed" << std::endl;
      return -1;
    }

    int sndbuf = 1024 * 1024;
    rc = zmq_setsockopt(puber, ZMQ_SNDBUF, &sndbuf, sizeof(sndbuf));
    if (rc < 0) {
      std::cout << "set sndbuf failed" << std::endl;
      return -1;
    }

    if (zmq_bind(puber, addr.c_str()) < 0) {
      std::cout << "pub bind failed: " << zmq_strerror(errno) << addr
                << std::endl;
      return -1;
    }

	  pub_inite = true;
    usleep(150000);
    return 0;
  }

  std::string file_path;
  bool pub_inite = false;
  void* context;
  void* puber;
};

class Push {
 public:
  Push(std::string filepath) {
    if (filepath == "") {
      std::cout << "no zmq ipc file input" << std::endl;
    }
    file_path = filepath;
    if ((context = zmq_ctx_new()) == NULL) {
      std::cout << "pub context init failed" << std::endl;
    }
    // InitPub();
  }
  ~Push() {
    zmq_close(pusher);
    zmq_ctx_destroy(context);
  }
  int SetSchedulePolicyPriority(int policy, int prioority) {
    if (context) {
      zmq_ctx_set(context, ZMQ_THREAD_SCHED_POLICY, policy);
      zmq_ctx_set(context, ZMQ_THREAD_PRIORITY, prioority);
    } else {
      return -1;
    }
    return 0;
  }

  int InitPush() {
    std::string addr = "";
    if (file_path.find("tcp") == std::string::npos) {
      if (access(file_path.c_str(), F_OK) == -1) {
        if (creat(file_path.c_str(), 0755) < 0) {
          std::cout << "create file failed" << std::endl;
          return -1;
        }
      }
      addr = "ipc://" + file_path;
    } else {
      addr = file_path;
    }

    if ((pusher = zmq_socket(context, ZMQ_PUSH)) == NULL) {
      std::cout << " pub socket init failed" << std::endl;
      return -1;
    }
    int hwm = 100;
    int rc = zmq_setsockopt(pusher, ZMQ_SNDHWM, &hwm, sizeof(hwm));
    if (rc < 0) {
      std::cout << "set sndhwm failed" << std::endl;
      return -1;
    }

    int linger = 1000;
    rc = zmq_setsockopt(pusher, ZMQ_LINGER, &linger, sizeof(linger));
    if (rc < 0) {
      std::cout << "set linger failed" << std::endl;
      return -1;
    }

    int sndbuf = 1024 * 1024;
    rc = zmq_setsockopt(pusher, ZMQ_SNDBUF, &sndbuf, sizeof(sndbuf));
    if (rc < 0) {
      std::cout << "set sndbuf failed" << std::endl;
      return -1;
    }
    int immediate = 1;
    rc = zmq_setsockopt(pusher, ZMQ_IMMEDIATE, &immediate, sizeof(immediate));
    if (rc < 0) {
      std::cout << "set immediate failed" << std::endl;
      return -1;
    }
    int sndtimeo = 1000;
    rc = zmq_setsockopt(pusher, ZMQ_SNDTIMEO, &sndtimeo, sizeof(sndtimeo));
    if (rc < 0) {
      std::cout << "set sndtimeo failed" << std::endl;
      return -1;
    }
    if (zmq_bind(pusher, addr.c_str()) < 0) {
      std::cout << "pub bind failed: " << zmq_strerror(errno) << addr
                << std::endl;
      return -1;
    }

    push_inite = true;
    usleep(150000);
    return 0;
  }

  std::string file_path;
  bool push_inite = false;
  void* context;
  void* pusher;
};

#endif  // CLASSSERVER_H