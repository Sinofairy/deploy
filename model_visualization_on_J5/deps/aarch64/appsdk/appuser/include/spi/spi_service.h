//
// Copyright 2020 Horizon      2021-07-23 14:11
//
#ifndef SPI_SERVICE_H
#define SPI_SERVICE_H

/******************************************************************************/
/*----------------------------------Includes----------------------------------*/
/******************************************************************************/
#include <sys/ipc.h>
#include <sys/shm.h>

#include "classserver.h"
#include "classclient.h"

/******************************************************************************/
/*-----------------------------------Macros-----------------------------------*/
/******************************************************************************/
#define SPI_SERVICE_MAX_LENGTH (4096u - 4u)

#define CAN_FILTER_EXT_NUM (50u)

#define SPI_TIMESYNC_SHM_KEY	(0x2e535049)	/* .SPI */
#define	SPI_TIMESYNC_SHM_SIZE	(4096)

/******************************************************************************/
/*--------------------------------Enumerations--------------------------------*/
/******************************************************************************/
enum {
	SERVICE_TYPE_SPI_SLAVE = 0,
	SERVICE_TYPE_SPI_MASTER = 1,
	SERVICE_TYPE_BIF_SPI = 2,
	SERVICE_TYPE_ETH = 3,
	SERVICE_TYPE_NUM
};

typedef enum {
  SPI_SERVICE_STATE_UNINIT = 0,
  SPI_SERVICE_STATE_INIT = 1,
  SPI_SERVICE_STATE_HANDSHAKE = 2,
  SPI_SERVICE_STATE_RUN = 3,
  SPI_SERVICE_STATE_DEINIT = 4,
  SPI_SERVICE_STATE_NUM
} SpiServiceType;

/******************************************************************************/
/*-----------------------------Data Structures--------------------------------*/
/******************************************************************************/

/**
 * @struct CanFrame_st
 * @param id: CAN id
 * @param dlc: CAN dlc
 * @param unused: unused
 * @param data: CAN data
 */
typedef struct {
  uint32_t id; /* CAN id */
  uint8_t dlc; /* CAN dlc */
  uint8_t unused[3];
  uint8_t data[8]; /* CAN data */
} CanFrame_st;

/**
 * @struct CanFDFrame_st
 * @param id: CAN id
 * @param dlc: CAN data length code
 * @param unused: unused
 * @param data: CAN data
 */
typedef struct {
  uint32_t id; /* CNA id */
  uint8_t dlc; /* CAN dlc */
  uint8_t unused[3];
  uint8_t data[64]; /* CAN data */
} CanFdFrame_st;

/**
 * @struct MsgHeader_st
 * @param type: Message type
 * @param length: data length
 * @param protocol_id: protocol id
 * @param timestamp: timestamp when sending
 */
typedef struct {
  uint16_t type;            /* Message type */
  uint16_t length;          /* Data length */
  uint32_t protocol_id;
  uint8_t timestamp[8];
} MsgHeader_st;

/**
 * @struct MsgTestMode_st
 * @param header: Message Header
 * @param sub_fun: sub function
 * @param unused: unused
 */
typedef struct {
  MsgHeader_st header;
  uint16_t sub_fun;
  uint8_t unused[2];
} MsgTestMode_st;

/**
 * @struct MsgTimeStamp_st
 * @param header: Message Header
 * @param timestamp: timestamp when sending
 * @param unused: unused
 */
typedef struct {
  MsgHeader_st header;
  long timestamp;
  uint8_t unused[1000];
} MsgTimestamp_st;

/**
 * @struct MsgOut_st
 * @param length: length of data
 * @param data: data of message
 */
typedef struct {
  int length;
  uint8_t data[SPI_SERVICE_MAX_LENGTH];
} MsgOut_st;

/**
 * @struct MsgGeneral_st
 * @param header: Message Header
 * @param data: data of message
 */
typedef struct {
  MsgHeader_st header;
  uint8_t data[SPI_SERVICE_MAX_LENGTH];
} MsgGeneral_st;

/**
 * @struct MsgSubSvc_st
 * @param header: Message Header
 * @param sub: sub function
 * @param data: data of message
 */
typedef struct {
  MsgHeader_st header;
  uint16_t sub;
  uint8_t data[SPI_SERVICE_MAX_LENGTH];
} MsgSubSvc_st;

typedef int (*svcFunc)(uint8_t *, int);
typedef int (*svcSubFunc)(uint8_t *, int);

/**
 * @struct SpiServiceTab_st
 * @param svcType: type of service
 * @param svcHandler: Handler of service
 */
typedef struct {
  uint16_t svcType;
  svcFunc svcHandler;
} SpiServiceTab_st;

/**
 * @struct SpiServiceSubTab_st
 * @param subType: type of sub function
 * @param subHandler: Handler of sub function
 */
typedef struct {
  uint16_t subType;
  svcSubFunc subHandler;
} SpiServiceSubTab_st;

/**
 * @struct MsgCANRaw_st
 * @param header: header of message
 * @param ts: timestamp when sending
 * @param data: CAN Raw data
 */
typedef struct {
  MsgHeader_st header;
  int64_t ts;
  uint8_t data[4000];
} MsgCANRaw_st;

/**
 * @struct CANHeader_st
 * @param time_stamp: timestamp when sending
 * @param counter: counter
 * @param frame_num: frame number
 */
typedef struct {
  int64_t time_stamp;
  uint8_t counter;
  uint8_t frame_num;
} CANHeader_st;

/**
 * @struct MsgCANFrames_st
 * @param header: CANheader
 * @param data: CAN frame data
 */
typedef struct {
  CANHeader_st header;
  uint8_t data[4000];
} MsgCANFrames_st;

/**
 * @struct DiagRead_st
 * @param sub: sub function
 * @param cmd: command
 * @param unused: unused
 */
typedef struct {
  uint16_t sub;
  uint8_t cmd[3];
  uint8_t unused[3]; 
} DiagRead_st;

/**
 * @struct MsgDiagRead_st
 * @param header: message header
 * @param diag_cmd: diagnose command
 */
typedef struct {
  MsgHeader_st header;
  DiagRead_st diag_cmd;
} MsgDiagRead_st;

/**
 * @struct MsgHandshakeSync_st
 * @param header: message header
 * @param code: code of Three-way Handshake
 * @param unused: unused
 */
typedef struct {
  MsgHeader_st header;
  uint8_t code;
  uint8_t unused[3];
} MsgHandshakeSync_st;

/**
 * @struct MsgHandshakeAck_st
 * @param header: message header
 * @param sync: sync symbol of Three-way Handshake
 * @param ack: fixed value 0x74
 * @param unused: unused
 */
typedef struct {
  MsgHeader_st header;
  uint8_t sync;
  uint8_t ack;
  uint8_t unused[2];
} MsgHandshakeAck_st;

// sub system
/**
 * @struct MsgAnswer_st
 * @param header: message header
 * @param answer: answer symbol of Three-way Handshake
 * @param unused: unused
 */
typedef struct {
    MsgHeader_st header;
    uint8_t      answer;
    uint8_t      unused[3];
} MsgAnswer_st;

/**
 * @struct MsgMgrState_st
 * @param header: message header
 * @param sub: sub function
 * @param unused: unused
 * @param state: state
 * @param unused: unused
 */
typedef struct {
    MsgHeader_st header;
    uint16_t     sub;
    uint8_t      unused[2];
    uint16_t     state;
    uint8_t      unused1[2];
} MsgMgrState_st;

/**
 * @struct MsgNOAact_st
 * @param header: message header
 * @param sub: sub function
 * @param ack: fixed value 0x74
 * @param unused: unused
 */
typedef struct {
    MsgHeader_st header;
    uint16_t     sub;
    uint8_t      ack;
    uint8_t      unused;
} MsgNOAact_st;

/**
 * @struct MsgMgrCaliResult_st
 * @param header: message header
 * @param sub: sub function
 * @param unused: unused
 * @param result: calibration result
 */
typedef struct {
    MsgHeader_st header;
    uint16_t     sub;
    uint8_t      unused[2];
    uint32_t     result;
} MsgMgrCaliResult_st;
// typedef struct{
//     float cpu_load;
//     int cpu_temp;
//     float  cpu_mem;
// } CPU_data_t;

/**
 * @struct CPU_data_t
 * @param cpu_load: state of cpu load
 * @param cpu_temp: state of cpu temp
 * @param cpu_mem: state of cpu memory
 * @param bpu0_load: state of bpu0 load
 * @param bpu1_load: state of bpu1 load
 * @param sensor_temp: state of sensor temp
 */
typedef struct{
    float cpu_load;
    int cpu_temp;
    float cpu_mem;
    int bpu0_load;
    int bpu1_load;
    int sensor_temp;
} CPU_data_t;

/**
 * @struct MsgMgrCPUState_st
 * @param header: message header
 * @param sub: sub function
 * @param unused: unused
 * @param cpu_data: cpu data
 */
typedef struct {
    MsgHeader_st header;
    uint16_t     sub;
    uint8_t     unused[2];
    CPU_data_t   cpu_data;
} MsgMgrCPUState_st;

/**
 * @struct SysMgrHeader_t
 * @param version: version of system
 * @param type: type
 * @param length: length of message
 */
// system head
typedef struct {
  uint16_t version;
  uint16_t type;
  uint16_t length;
} SysMgrHeader_t;

// PUB system message

/**
 * @struct SysMgrCPUState_st
 * @param header: header of message
 * @param filter: filter(QA)
 * @param cpu_data: cpu data
 */
typedef struct {
  SysMgrHeader_t header;
  uint8_t filter[6];
  CPU_data_t cpu_data;
} SysMgrCPUState_st;

/**
 * @struct SysMgrQA_st
 * @param header: header of message
 * @param filter: filter(QA)
 * @param question: question
 * @param unused: unused
 */
typedef struct {
    SysMgrHeader_t header;
    uint8_t  filter[2];
    uint8_t  question;
    uint8_t  unused;
} SysMgrQA_st;

/**
 * @struct SysMgrAns_st
 * @param filter: filter(QA)
 * @param answer: answer
 * @param unused: unused
 */
typedef struct {
    SysMgrHeader_t header;
    uint8_t  filter[2];
    uint8_t  answer;
    uint8_t  unused;
} SysMgrAns_st;

/**
 * @struct SysMgrState_st
 * @param filter: filter(QA)
 * @param state: state
 */
typedef struct {
    SysMgrHeader_t header;
    uint8_t  filter[6];
    uint16_t state;
} SysMgrState_st;

/**
 * @struct SysMgrActStatus_st
 * @param filter: filter(QA)
 * @param NOA_act_status: NOA act status
 */
typedef struct {
    SysMgrHeader_t header;
    uint8_t  filter[6];
    uint16_t NOA_act_status;
} SysMgrActStatus_st;

/**
 * @struct SysMgrDeviceId_st
 * @param filter: filter(QA)
 * @param device_id: device ID
 */
typedef struct {
    SysMgrHeader_t header;
    uint8_t filter[6];
    uint8_t device_id[10];
} SysMgrDeviceId_st;

/**
 * @struct SysMgrDeviceId_ack_st
 * @param filter: filter(QA)
 * @param device_id: device ID
 */

typedef struct {
    SysMgrHeader_t header;
    uint8_t filter[6];
    uint8_t mask;
    uint8_t type;
} SysMgrDeviceId_ack_st;
/**
 * @struct SysMgrCaliResult_st
 * @param filter: filter(QA)
 * @param result: result
 */
typedef struct {
    SysMgrHeader_t header;
    uint8_t  filter[6];
    uint16_t result;
} SysMgrCaliResult_st;

/**
 * @struct ResetInfo_st
 * @param filter: filter(QA)
 * @param cmd: command
 */
typedef struct {
  SysMgrHeader_t header;
  uint8_t  filter[6];
  uint16_t cmd;
} ResetInfo_st;

/**
 * @struct CalibExtrinInfo_st
 * @param camera_x: camera in label x
 * @param camera_y: camera in label y
 * @param camera_z: camera in label z
 * @param pitch: pitch
 * @param yaw: yaw
 * @param roll: roll
 * @param update: updata
 * @param unused:unused
 */
typedef struct {
  float camera_x;
  float camera_y;
  float camera_z;
  float pitch;
  float yaw;
  float roll;
  uint8_t update;
  uint8_t unused[3];
} CalibExtrinInfo_st;

/**
 * @struct SysMgrExtrin_st
 * @param filter: filter(QA)
 * @param info: information
 */
typedef struct {
    SysMgrHeader_t header;
    uint8_t  filter[6];
    CalibExtrinInfo_st info;
} SysMgrExtrin_st;

/**
 * @struct CalibExtrin_st
 * @param state: state
 * @param param: paramter
 */
typedef struct {
  uint32_t state;
  float param[7];
} CalibExtrin_st;

/**
 * @struct SocCalibExtrin_st
 * @param type: type
 * @param state: state
 * @param param: paramter
 */
typedef struct {
  int32_t type;
  int32_t state;
  float param[7];
} SocCalibExtrin_st;

/**
 * @struct CalibExtrinMsg_st
 * @param header: message header
 * @param sub: sub function
 * @param unused: unused
 * @param info: information
 */
typedef struct {
  MsgHeader_st header;
  uint16_t  sub;
  uint8_t unused[2];
  CalibExtrin_st info;
} CalibExtrinMsg_st;

// sub reset info
/**
 * @struct MsgReset_st
 * @param header: message header
 * @param sub: sub function
 * @param unused: unused
 * @param param: paramter
 */
typedef struct {
  MsgHeader_st header;
  uint16_t sub;
  uint8_t param;
  uint8_t unused;
} MsgReset_st;

// sub pwm monitor info
/**
 * @struct MsgPwnMon_st
 * @param header: message header
 * @param sub: sub function
 * @param unused: unused
 */
typedef struct {
  MsgHeader_st header;
  uint16_t sub;
  uint8_t unused[2];
} MsgPwmMon_st;



typedef struct {
  uint8_t can_channel; // 0 ~ 5
  uint8_t verbose;
  uint8_t unsed;
  uint8_t can_num; // max 50
  uint32_t can_id[CAN_FILTER_EXT_NUM]; // max 50
} CanFilterExt_st;

typedef struct {
  MsgHeader_st header;
  uint16_t     sub;
  uint8_t      unused[2];
  CanFilterExt_st info;
} MsgCanFilterExt_st;

typedef struct {
    uint32_t bookend1;
	uint32_t data_len;
    uint8_t data[SPI_TIMESYNC_SHM_SIZE - sizeof(uint32_t) * 3];
    uint32_t bookend2;
} SpiTimesyncData_st;

/******************************************************************************/
/*------------------------------Global variables------------------------------*/
/******************************************************************************/
/**
 * @brief Service manager class
 */
class ServiceManager {
 public:
  ServiceManager(void) {}
  ~ServiceManager(void) {}
  virtual int service_manager_state_machine(void);
  int spi_service_set_state(SpiServiceType state);
  int spi_service_get_state(SpiServiceType * state);
  SpiServiceType spi_service_state;
};

/**
 * @brief SPI Service slave manager class
 */
class ServiceManagerSpiSlave: public ServiceManager {
 public:
  ServiceManagerSpiSlave(int type) {
    type_ = type;
  }
  ~ServiceManagerSpiSlave(void) {}
  virtual int service_manager_state_machine(void);
  int spi_service_handshake_process(void);
  int type_;
};

/**
 * @brief publish CAN input message class
 */
class PubCANInput: public Pub {
 public:
  PubCANInput(std::string filepath):Pub(filepath) {}
  ~PubCANInput(void) {}
  int IpcCANInputPub(uint8_t * data, int length);
};

/**
 * @brief Subscribe CAN input message class
 */
class SubCANInput: public Sub{
 public:
  SubCANInput(std::string filepath):Sub(filepath) {
  }
  ~SubCANInput() {}
  int IpcCANInputSub(void);
};

/**
 * @brief pull CAN input data
 */
class PullCANInput: public Pull {
 public:
  explicit PullCANInput(std::string filepath):Pull(filepath) {
  }
  ~PullCANInput() {}
  int IpcCANInputPull(void);
};

class PullCANFilter: public Pull {
 public:
  explicit PullCANFilter(std::string filepath):Pull(filepath) {
  }
  ~PullCANFilter() {}
  int IpcCANFilterPull(void);
};

class SubCANOutput: public Sub{
 public:
  SubCANOutput(std::string filepath):Sub(filepath) {
  }
  ~SubCANOutput() {}
  int IpcCANOutputSub(void);
  int IpcCANOutputExtraSub(void);
};

/**
 * @brief subscribe calibration output data
 */
class SubCalibOutput: public Sub{
 public:
  SubCalibOutput(std::string filepath):Sub(filepath) {
  }
  ~SubCalibOutput() {}
  int IpcCalibOutputSub(void);
};

/**
 * @brief pull reset output data
 */
class PullResetOutput: public Pull{
 public: 
  explicit PullResetOutput(std::string filepath): Pull(filepath) {
  }
  ~PullResetOutput() {}
  int IpcResetOutputPull(void);
};

/**
 * @brief publish diagnose input data
 */
class PubDiagIn: public Pub {
 public:
  PubDiagIn(std::string filepath):Pub(filepath) {
    
  }
  ~PubDiagIn() {}
  int IpcDiagInPub(uint8_t * data, int length);
};

/**
 * @brief 
 */
class PubGps_subimu_In: public Pub {
 public:
  explicit PubGps_subimu_In(std::string filepath):Pub(filepath) {
  }
  ~PubGps_subimu_In() {}
  // int IpcGps_subimu_InPub(uint8_t * data, int length);
  void senddata(uint8_t * data, int length);
  /**
    * @brief calculate Checksum.
    * @param[in] data: data wait for publishing.
    * @param[in] size: length of data.
    * @param[in] ck_a:
    * @param[in] ck_b:
    */
  void calculateChecksum_(const uint8_t *data,
                               uint32_t size,
                               uint8_t &ck_a,
                               uint8_t &ck_b);
};
class PubGpsIn: public Pub {
 public:
  explicit PubGpsIn(std::string filepath):Pub(filepath) {
    // pubGps_subimu_In = new PubGps_subimu_In("/tmp/gps_subimu_sub.ipc");
    pubGps_subimu_In = new PubGps_subimu_In("tcp://127.0.0.1:3333");
  }
  ~PubGpsIn() {}
  int IpcGpsInPub(uint8_t * data, int length);
  void senddata(uint8_t * data, int length);
  uint8_t CharToHex(char bHex);
  bool calculateChecksum_(const uint8_t *data,
                                uint32_t size,
                                uint8_t &ck_a,
                                uint8_t &index);

 private:
  enum {
    HB_UNPACK_TYPE_NONE,
    HB_UNPACK_TYPE_GPS,
    HB_UNPACK_TYPE_IMU,
    HB_UNPACK_TYPE_NUM
  };
  uint8_t hb_data[1024] = {0};
  uint16_t hb_index = 0;
  uint16_t hb_len = 0;
  uint8_t hb_unpack_type = 0;
  bool imu_type;
  PubGps_subimu_In* pubGps_subimu_In;
};

/**
 * @brief publish IMU input data
 */
class PubImuIn: public Pub {
 public:
  PubImuIn(std::string filepath):Pub(filepath) {
  }
  ~PubImuIn() {}
  int IpcImuInPub(uint8_t * data, int length);
};

/**
 * @brief subscribe diagnose output data
 */
class SubDiagOut: public Sub{
 public:
  SubDiagOut(std::string filepath):Sub(filepath) {
  }
  ~SubDiagOut() {}
  int IpcDiagOutSub(void);
};

/**
 * @brief publish QA input data
 */
class PubQAIn: public Pub {
 public:
  PubQAIn(std::string filepath):Pub(filepath) {
  }
  ~PubQAIn() {}
  int IpcQAInPub(uint8_t * data, int length);
  int IpcSocStateInPub(uint8_t * data, int length);
  int IpcCalibExtrinInPub(uint8_t * data, int length);
  int IpcIdDataInPub(uint8_t * data, int length);
  int IpcNOAActStatusPub(uint8_t * data, int length);
};

/**
 * @brief subscribe QA output data
 */
class SubQAOut: public Sub{
 public:
  SubQAOut(std::string filepath):Sub(filepath) {
  }
  ~SubQAOut() {}
  int IpcQAOutSub(void);
};

/**
 * @brief publish timesync input data
 */
class PushTimesyncIn: public Push {
 public:
  PushTimesyncIn(std::string filepath):Push(filepath) {
    IpcTimesyncShmInit();
  }
  ~PushTimesyncIn() {
	shmdt(shmseg);
	shmctl(shmid, IPC_RMID, 0);
  }
  int IpcTimesyncInPush(uint8_t * data, int length);
  int IpcTimesyncShmInit(void);
  int IpcTimesyncShmPush(uint8_t * data, int length);

  int shmid;
  void *shmseg;
};

/**
 * @brief publish utity input data
 */
class PubUtityIn: public Pub {
 public:
  PubUtityIn(std::string filepath):Pub(filepath) {
    
  }
  ~PubUtityIn() {}
  int IpcUtityInPub(uint8_t * data, int length);
};

/**
 * @brief subscribe utity output data
 */
class SubUtityOut: public Sub{
 public:
  SubUtityOut(std::string filepath):Sub(filepath) {
  }
  ~SubUtityOut() {}
  int IpcUtityOutSub(void);
};

/**
 * @brief publish testmode input data
 */
class PubTestModeIn: public Pub {
 public:
  PubTestModeIn(std::string filepath):Pub(filepath) {
  }
  ~PubTestModeIn() {}
  int IpcTestModeInPub(uint8_t * data, int length);
};

/**
 * @brief subscribe testmode output data
 */
class SubTestModeOut: public Sub{
 public:
  SubTestModeOut(std::string filepath):Sub(filepath) {
  }
  ~SubTestModeOut() {}
  int IpcTestModeOutSub(void);
};
/******************************************************************************/
/*-------------------------Function Prototypes--------------------------------*/
/******************************************************************************/

#endif  // SPI_SERVICE_H
