//
// Copyright 2016 Horizon Robotics.
// Created by Lisen Mu on 7/20/16.
//

#ifndef HOBOT_HOBOT_H_
#define HOBOT_HOBOT_H_

#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <deque>
#include <tuple>
#include <utility>
#include <mutex>
#include <functional>

#ifdef HR_WIN
#ifdef HOBOT_DLL_EXPORTS
#ifdef HOBOT_EXPORTS
#define HOBOT_EXPORT __declspec(dllexport)
#else
#define HOBOT_EXPORT __declspec(dllimport)
#endif
#else
#define HOBOT_EXPORT
#endif
#elif HR_POSIX
#define HOBOT_EXPORT
#endif

#ifndef INT32_MAX
#define INT32_MAX 2147483647
#endif

#define FORWARD_FUNCTION_SIGNATURE_(module_name, forward_index) \
  void Forward##forward_index(const hobot::MessageLists &input, \
  hobot::Workflow *workflow, hobot::spRunContext context)

#define WRAPPER_CLASS_(module_name, forward_index) \
  module_name##_##Forward##forward_index##_Wrapper

#define WRAPPER_INSTANCE_(module_name, forward_index) \
  module_name##_##Forward##forward_index##_Wrapper_Instance

/**
 * forward declare macro
 */
#define FORWARD_DEFAULT_THREAD_STACK_SIZE 0x200000

#define FORWARD_DECLARE(module_name, forward_index) \
        FORWARD_DECLARE_STACK_SIZE(module_name, forward_index, \
                                   FORWARD_DEFAULT_THREAD_STACK_SIZE)

#define FORWARD_DECLARE_STACK_SIZE(module_name, forward_index, \
                                                stack_size_default) \
  friend class WRAPPER_CLASS_(module_name, forward_index); \
  class WRAPPER_CLASS_(module_name, forward_index) \
  : public hobot::ForwardWrapper {  \
    public:                                                 \
      WRAPPER_CLASS_(module_name, forward_index)(module_name *module, \
                                      size_t stack_size = stack_size_default) \
       : hobot::ForwardWrapper(), module(module) { \
           stack_size_ = stack_size; \
           run_time_ = 0; \
           run_time_max_ = 0; \
           run_time_min_ = INT64_MAX; \
           ts_ = 0; \
           start_ts_ = 0; \
           run_count_ = 0; \
           module->RegisterForwardWrapper(this, forward_index);    \
      }  \
      void Forward(const hobot::MessageLists &input, \
                     hobot::Workflow *workflow, \
                     hobot::spRunContext context) { \
          std::string module_str = module->GetFullClassName(); \
          if (module->checkforward_ && module_str != "") { \
            module->checkforward_(workflow, context, module, module_str, \
             forward_index); \
          } \
          module->Forward##forward_index(input, workflow, context) ; \
              }                            \
    private: \
      module_name *module;                                        \
  };\
  WRAPPER_CLASS_(module_name, forward_index) \
     WRAPPER_INSTANCE_(module_name, forward_index) =  \
          WRAPPER_CLASS_(module_name, forward_index)(this);  \
  FORWARD_FUNCTION_SIGNATURE_(module_name, forward_index)


/**
 * forward define macro
 */
#define FORWARD_DEFINE(module_name, forward_index) \
     void module_name::Forward##forward_index \
        (const hobot::MessageLists &input, \
        hobot::Workflow *workflow, hobot::spRunContext context)


/**
 * config update listener define macro
 */
#define LISTENING_CLASS_(module_name, key_name) \
  module_name##_##key_name##_Wrapper

#define LISTENING_INSTANCE_(module_name, key_name) \
  module_name##_##key_name##_Wrapper_Instance

#define LISTENING_FUNCTION_SIGNATURE_(module_name, key_name) \
  void On##key_name##Update(const std::string& key)

/**
 * config update listener declare macro
 * Recommend way is to place this at the end inside your class declaration.
 * This is because the "private" keyword inside, which would cause members
 * after this macro to be private. User should modify your code accordingly,
 * if this is placed in the midle of yoru class declaration.
 */
#define LISTENING_CONFIG_DECLARE(module_name, key_name) \
  friend class LISTENING_CLASS_(module_name, key_name); \
  class LISTENING_CLASS_(module_name, key_name)  { \
  public:    \
      LISTENING_CLASS_(module_name, key_name)(module_name* module) { \
        module->configUpdateFns_[#key_name] = [module] \
          (const std::string &key) { \
          module->On##key_name##Update(key); \
        };  \
      } \
  }; \
  private: \
    LISTENING_CLASS_(module_name, key_name) \
      LISTENING_INSTANCE_(module_name, key_name) =  \
        LISTENING_CLASS_(module_name, key_name)(this); \
    LISTENING_FUNCTION_SIGNATURE_(module_name, key_name)

namespace hobot {

class Config;

class Module;

class ForwardWrapper;

class InputModule;

class Message;

class Expression;

class Workflow;

class LinkBuilder;

class Link;

class RunContext;

class ConfigExt;

typedef std::shared_ptr<LinkBuilder> spLinkBuilder;

typedef std::shared_ptr<Message> spMessage;

typedef std::shared_ptr<RunContext> spRunContext;

typedef std::deque<spMessage> MessageList;

typedef std::vector<MessageList *> MessageLists;

typedef std::shared_ptr<Config> spConfig;

typedef std::shared_ptr<ConfigExt> spConfigExt;

class Broker;

typedef std::shared_ptr<Broker> spBroker;

class BrokerStatus;

typedef std::shared_ptr<BrokerStatus> spBrokerStatus;

class Client;

typedef std::shared_ptr<Client> spClient;

typedef std::vector<spClient> ClientList;

class MsgHandler;

typedef std::shared_ptr<MsgHandler> spMsgHandler;

// pub-sub client subscription message callback function
typedef std::function<bool(MsgHandler *, const std::string &, spMessage)>
    MsgCallback;

enum class ActionType { PUB, SUB };

class HOBOT_EXPORT Engine {
 public:
  /**
   * creates a new Engine.
   * @return created engine
   */
  static Engine *NewInstance();

  /**
  * creates a new Thread Save Engine.
  * If workflow is fixed(no dynamic workflow change), NewInstance() is enough.
  * Otherwise, Thread Safe engine is Required.
  * @return created engine
  */
  static Engine *NewInstanceThreadSafe();

  /**
   * creates a new workflow to be executed on this engine.
   * @return created workflow
   */
  virtual Workflow *NewWorkflow() = 0;

  /**
   * specify forward runs on which thread.
   * forward{index} may run on the thread which forward{random index} runs on by default
   * Engine maintains a inner threadpool on which forward would run.
   */
  virtual bool ExecuteOnThread(Module *module,
                               int forward_index,
                               int thread_index) = 0;

  /**
   * Sets the cpu affinity of thread.
   */
  virtual bool SetAffinity(int thread_index, int core_id) = 0;

  virtual int GetThreadIdx(Module *module, int forward_index) = 0;

  virtual bool ThreadRun(int thread_index) = 0;

  virtual void ThreadStop(int thread_index) = 0;

  /**
   * Enable Monitor function (optional)
   * @param monitor_period : the peroid_time(ms) monitor event thread
   * @param delay_start : the monitor thread start running after delay_start ms
   */
  virtual bool EnableMonitor(int32_t monitor_period = 0,
                             int32_t delay_start = 0) {
    return (monitor_period >= 0 && delay_start >= 0) ? true : false;
  }

  virtual ~Engine() { }
};

class RunObserver;

typedef struct {
  std::string topic;
  int slot_idx;
  int forward_idx;  // only module sub topic has this item
} ModuleTopic;

/**
 * A Directed Graph specifying executing dependencies of Modules,
 * Forming a more complicated functionality than a single Module.
 */
class HOBOT_EXPORT Workflow {
 public:
  /**
   * create a LinkBuilder object to create links from module `src`.
   * workflow->From(a)->To(b) to construct a link a -> b.
   * @param src source Module of the links to be created
   * @param output_slot Module's output slot index
   * @return LinkBuilder to build links
   */
  virtual spLinkBuilder From(Module *src, int output_slot = 0) = 0;


  /**
  * Generate Task .
  * @param outputs collection of Modules whose output would be gathered by RunObserver
  * @param ob callback handler to gather output for outputs
  * @param message
  */
  virtual spRunContext Run(std::vector<std::pair<Module *, int>> outputs,
                           RunObserver *ob = nullptr,
                           Message *message = nullptr) = 0;

  /**
   * Run the graph. once. function `Run` above is strongly recommended
   * @param outputs collection of Modules whose output would be gathered by RunObserver
   * @param inputs input messages, map from InputModule to Message
   * @param ob callback handler to gather output for outputs
   * @code return init code
   * @module return the module whitch init failed
   * @param message
   */
  virtual spRunContext Run(std::vector<std::pair<Module *, int>> outputs,
                           std::vector<std::tuple<Module *,
                                                  int,
                                                  spMessage>> inputs,
                           RunObserver *ob = nullptr,
                           int *code = nullptr,
                           Module **module = nullptr,
                           Message *message = nullptr) = 0;

  /**
   * Feed and run the graph.
   * @param run_context
   * @param module
   * @param forward_index
   * @param message
   */
  virtual void Feed(spRunContext run_context,
                    Module *module,
                    int forward_index,
                    spMessage message) = 0;

  /**
  * Feed and run the graph.
  * @param run_context
  * @param module
  * @param forward_index
  * @param intput slot index
  * @param message
  */
  virtual void Feed(spRunContext run_context,
                    Module *module,
                    int forward_index,
                    int input_slot_index,
                    spMessage message) = 0;

  /**
   * Reset , clear all messages in the flow and reset modules
   */
  virtual void Reset() = 0;


  /**
   * get run condition of module in current workflow.
   * @param module module to get
   * @param forward_index forward index
   * @return the run condition of module
   */
  virtual Expression *ConditionOf(Module *module, int forward_index) = 0;

  /**
   * set run condition of module in current workflow.
   * Default run condition is AllExp:
   * every input has at least 1 message;
   * each input will fetch 1 message.
   * @param module module
   * @param condition condition
   * @param forward_index forward index
   */
  virtual void SetCondition(Module *module,
                            Expression *condition,
                            int forward_index) = 0;

  /**
   * set io flusher of module in current workflow.
   * Default io flusher is inactive.
   * @param module module
   * @param millisec io flusher will flush all input data every millisec
   * @param forward_index forward index
   */
  virtual void SetFlusher(Module *module,
                            int millisec,
                            int forward_index) = 0;

  /**
  * set module forward execution timeout in millisecond.
  * Observer->OnTimeout() will be triggered if timeout
  * Default timeout behavior is inactive.
  * @param module module
  * @param millisecond for timeout
  * @param forward_index forward index
  */
  virtual void SetExecTimeout(Module *module,
                              int millisec,
                              int forward_index,
                              spMessage err_msg = nullptr) = 0;

  /**
  * set module forward expect execution period time in millisecond
  * monitor thread will send one error msg Observer->OnError()
  * when has elapse millsec time since last forward run finish.
  * Default period time is inactive.
  * @param module module
  * @param millisecond for cycle time
  * @param forward_index forward index
  */
  virtual void SetExecPeriodTime(Module *module,
                              int millisec,
                              int forward_index,
                              spMessage err_msg = nullptr) = 0;

  /**
   * should be called within Module's Forward(),
   * to produce Module's output message
   * @param from module
   * @param output_slot Module's output slot index
   * @param output message produced
   * @param context the RunContext
   */
  virtual void Return(Module *from,
                      int output_slot,
                      spMessage output,
                      spRunContext context) = 0;


  /**
   * may be called within Module's Forward or Init on  error condition
   * @param from
   * @param forward_index
   * @param err error message produced
   * @param context the RunContext
   */
  virtual void Error(Module *from,
                     int forward_index,
                     spMessage err,
                     spRunContext context) = 0;

  /**
  * may be called within Module's Forward or Init on error condition
  * and not blocking any thread
  * @param from
  * @param forward_index
  * @param err error message produced
  * @param context the RunContext
  */
  virtual void ErrorNoBlock(Module *from,
                            int forward_index,
                            spMessage err,
                            spRunContext context) = 0;

  /**
   * could be called within Module's Forward(),
   * notifying another Forward() in the future
   * @param from module
   * @param forward_index forward index
   * @param input input data of next Forward()
   * @param context the RunContext
   * @param millisec delay of next Forward(), in milliseconds, from now
   */
  virtual void Reschedule(Module *from, int forward_index,
                          const MessageLists &input,
                          const spRunContext &context, int millisec) = 0;

  virtual ~Workflow() { }

  /**
  * could be called within Module's Forward(),
  * notifying another Forward() in the future
  * @param module module ptr
  * @param forward_index forward index
  * @param input_slot module's input slot of specific forward index
   * @return if Module's input_slot@forward_index is linked
  */
  virtual bool CheckInputLink(Module *module,
                              int forward_index, int input_slot) = 0;

  /**
  * could be called within Module's Forward(),
  * notifying another Forward() in the future
  * @param module module ptr
  * @param forward_index forward index
  * @param output_slot module's output slot
  * @return if Module's output_slot is linked
  */
  virtual bool CheckOutputLink(Module *module, int output_slot) = 0;


  /**
  * manually consume link limit counter
  * @param module module ptr
  * @param forward_index forward index
  * @param input_slot module's input slot
  * @param size limit size
  */
  virtual void ConsumeLinkLimit(spRunContext run_context, Module *module,
                                int forward_index, int input_slot,
                                int size) = 0;

  /**
  * manually release link limit counter
  * @param module module ptr
  * @param forward_index forward index
  * @param input_slot module's input slot
  * @param size limit size
  */
  virtual void ReleaseLinkLimit(spRunContext run_context, Module *module,
                                int forward_index, int input_slot,
                                int size) = 0;
  /**
  * set output slots and forward index binding
  * @param module module ptr
  * @param forward_index forward index
  * @param output_slots bindings(unbind if output_slots is empty)
  */
  virtual void SetOutputSlotBinding(Module *module, int forward_index,
                                    const std::vector<int> &output_slots) = 0;

  /**
  * set thread name
  * @param module module ptr
  * @param forward_index forward index
  * @param thread_name thread name
  * @return -2: OverLength, -1: NameFail, 0: NameSuccess
  */
  virtual int SetThreadName(Module *module, int forward_index,
                            std::string thread_name) = 0;

  /**
   * fetch a Broker based on this workflow
   * @param none
   * @return the broker based on this workflow
   */
  virtual spBroker FetchBroker() = 0;

  /**
   * fetch related topic list according to the workflow module, according to the
   * module's GenSub/GenPub settings. This interface is usually called by outer
   * program.
   * @param module  a module pointer within this workflow
   * @param type  action type, one of [PUB, SUB]
   * @return related topic list.
   */
  virtual std::vector<ModuleTopic> TopicList(Module *module,
                                             ActionType type) = 0;

  /**
   * fetch related topic list according to the workflow module.  This interface
   * is usually called by workflow creator.
   * @param module  a module pointer within this workflow
   * @param type  action type, one of [PUB, SUB]
   * @return related topic list.
   */
  virtual std::string SubTopic(Module *module, int forward_idx,
                               int input_slot) = 0;

  /**
   * fetch related topic list according to the workflow module.This interface
   * is usually called by workflow creator.
   * @param module  a module pointer within this workflow
   * @param type  action type, one of [PUB, SUB]
   * @return related topic list.
   */
  virtual std::string PubTopic(Module *module, int output_slot) = 0;
};

/**
 * Callback interface of Workflow::Run()
 * If user wishes to receive results of Workflow::Run(),
 * they should provide a instance of subclass of RunObserver to Run().
 */
class HOBOT_EXPORT  RunObserver {
 public:
  /**
   * call back to receive output from run
   * @param from
   * @param forward_index
   * @param output the returned message, same order specified
   * with Run()
   */
  virtual void OnResult(Module *from, int forward_index, spMessage output) = 0;

  /**
   *  callback to handle error from run ( Init and Forward included )
   *  @param from
   *  @param forward_index
   *  @param err
   */
  virtual void OnError(Module * /*from*/, int /*forward_index*/,
                       spMessage /*err*/) { }

  /**
  *  callback to handle timeout module's forward execution
  *  @param from
  *  @param forward_index
  */
  virtual void OnExecTimeout(Module * /*from*/,
                             int /*forward_index*/, int64_t /*run_time*/) { }

  virtual ~RunObserver() { }
};

class HOBOT_EXPORT  LinkBuilder {
 public:
  /**
   * Link links module A's output to module B's input slot
   * Every module has exact one output;
   * can have zero or more input slots, indexing from 0 to N.
   * @param dest destination module
   * @param index index if input slot of `dest`
   * @param forward_index forward index
   */
  virtual Link *To(Module *dest,
                   int input_index = 0,
                   int forward_index = 0) = 0;
  // do we need another start_size to control
  // minimum requirement for output buffer?

  virtual ~LinkBuilder() { }
};

class HOBOT_EXPORT  Link {
 public:
  /**
   * Limit link output buffer size
   */
  virtual void Limit(int buffer_size = INT32_MAX) = 0;

  virtual ~Link() { }
};

typedef union HOBOT_EXPORT _Variant32 {
  _Variant32(): valueint(0) {}
  _Variant32(int _value): valueint(_value) {}
  _Variant32(float _value): valuefloat(_value) {}
  _Variant32(bool _value): valuebool(_value) {}
  _Variant32(const _Variant32 &v): valueint(v.valueint) {}

  // char int long
  int valueint;
  float valuefloat;
  bool valuebool;
} Variant32;

typedef union HOBOT_EXPORT _Variant64 {
  _Variant64() {}
  _Variant64(int64_t _value): valuelong(_value) {}
  _Variant64(double _value): valuedouble(_value) {}

  // char int long
  int64_t valuelong;
  double valuedouble;
} Variant64;

class HOBOT_EXPORT Config {
 public:
  static spConfig LoadConfig(const std::string &filename);
  static spConfig LoadStringConfig(const std::string &config_string);
  Config() { }

  template<typename T>
  void SetParams(const std::string &key, T value);

  int GetIntValue(const std::string &key, int default_value = 0);

  float GetFloatValue(const std::string &key, float default_value = 0.0f);

  bool GetBoolValue(const std::string &key, bool default_value = false);

  int64_t GetLongValue(const std::string &key, int64_t default_value = 0);

  double GetDoubleValue(const std::string &key, double default_value = 0.0);

  const std::string & GetSTDStringValue(const std::string &key,
                                     const std::string &default_value = "");

  const char * GetStringValue(const std::string &key,
                              const char * default_value = "");

  // will insert a new empty sub config if sub_configs_[name] do not exist
  Config *GetSubConfig(const std::string &name);

  void AddSubConfig(const std::string &name, spConfig config);

  bool IsEmpty(void);

  Config &operator=(const Config &other);

 private:
  template<typename T>
  const T &GetParamValue(const std::string &key, const T &default_vaule);

  typedef std::map<std::string, Variant32> MapVariant32;
  typedef std::map<std::string, Variant64> MapVariant64;
  typedef std::map<std::string, std::string> MapString;
  typedef std::map<std::string, spConfig> MapSpConfig;

  MapVariant32 values32_;

  MapVariant64 values64_;

  MapString values_string_;

  MapSpConfig sub_configs_;

  std::mutex config_mutex_;
};

class HOBOT_EXPORT ConfigExt {
 public:
  ConfigExt() { }

  explicit ConfigExt(spMessage handle);

  bool LoadConfig(const std::string &filename);

  bool LoadStringConfig(const std::string &config_string);

  bool Release();

  bool Save2File(std::string another_file = "");

  std::string Save2String() const;

  bool IsMember(const std::string &key);

  bool IsEmpty(void);

  int GetIntValue(const std::string &key, int default_value = 0);

  float GetFloatValue(const std::string &key, float default_value = 0.0f);

  bool GetBoolValue(const std::string &key, bool default_value = false);

  int64_t GetLongValue(const std::string &key, int64_t default_value = 0);

  double GetDoubleValue(const std::string &key, double default_value = 0.0);

  const std::string GetSTDStringValue(const std::string &key,
                                      const std::string &default_value = "");

  spConfigExt GetSubConfig(const std::string &name);

  spConfigExt GetArray(const std::string &arrayname);

  int GetArraySize();

  bool SetIntValue(const std::string& key, int value);

  bool SetFloatValue(const std::string& key, float value);

  bool SetBoolValue(const std::string& key, bool value);

  bool SetLongValue(const std::string& key, int64_t value);

  bool SetDoubleValue(const std::string& key, double value);

  bool SetSTDStringValue(const std::string& key, std::string& value);

  bool DelItem(const std::string& key);

  spConfigExt operator[](int index);

 private:
  spMessage jhanle_;
};

struct ThreadIDWrapper;

class ForwardWrapper {
 public:
  virtual void Forward(const MessageLists &input,
                       Workflow *workflow,
                       spRunContext context) = 0;

  virtual ~ForwardWrapper();

  size_t GetStackSize() {
    return stack_size_;
  }

  int64_t GetRunTime() {
    return run_time_;
  }

  int64_t GetRunTimeMax() {
    return run_time_max_;
  }

  int64_t GetRunTimeMin() {
    return run_time_min_;
  }

  int64_t GetUpdateTimeStamp() {
    return ts_;
  }

  void SetUpdateTimeStamp(uint64_t time_in_micro) {
    ts_ = static_cast<int64_t>(time_in_micro);
  }

  void SetRunTime(uint64_t new_run_time) {
    run_time_ = static_cast<int64_t>(new_run_time);
    // update run_time_max_
    if (run_time_ > run_time_max_) {
      run_time_max_ = run_time_;
    }
    // update run_time_min_
    if (run_time_ > 0 && run_time_ < run_time_min_) {
      run_time_min_ = run_time_;
    }
  }

  void SetStartTime(int64_t now) {
    start_ts_ = now;
  }

  int64_t GetStartTime(void) const {
    return start_ts_;
  }

  void SetRunCount(uint64_t run_count) {
    run_count_ = run_count;
  }

  void IncreaseRunCount() {
    run_count_++;
  }

  uint64_t GetRunCount() const {
    return run_count_;
  }

  ThreadIDWrapper *GetThreadIDWrapper() const { return thread_id_wrapper_; }
  void SetThreadIDWrapper(const ThreadIDWrapper& thread_id_wrapper);

 protected:
  size_t stack_size_;
  int64_t run_time_;
  int64_t run_time_max_;
  int64_t run_time_min_;
  int64_t ts_;
  int64_t start_ts_;  // 0 means not running
  ThreadIDWrapper* thread_id_wrapper_ = nullptr;
  uint64_t run_count_;
};

typedef std::function<void(const std::string &)> fnConfigUpdate;
typedef std::function<void(Workflow*, spRunContext, Module*,
 std::string, int)> CheckForward;
class HOBOT_EXPORT Module {
 public:
  explicit Module(std::string instance_name = "", std::string class_name = "")
      : inited_(false),
        class_name_(class_name),
        instance_name_(instance_name) { }

  virtual std::string GetFullClassName();

  /**
   *
   * @return the Full Name of this instance; format/content of the full name
   * is determined by implementation, but usually would contain Class Name and
   * Instance ID.
   */
  virtual std::string GetFullInstanceName();

  virtual void SetInstanceName(const std::string &instance_name) {
    this->instance_name_ = instance_name;
  }

  /**
   * Init will be called exactly once, for this module,
   * in each <engine, workflow> composition
   */
  virtual int Init(RunContext * /*context*/) = 0;

  void UpdateConfig(spConfig config);

  void UpdateConfig(const Config &config);

  template<typename T>
  void UpdateConfig(const std::string &key, T value);

  template<typename T>
  void UpdateConfig(const std::vector<std::pair<std::string, T>>& params);

  Config *GetConfig();

  virtual void OnConfigUpdate();

  virtual void Reset() = 0;

  size_t GetForwardCount() {
    return forward_wrappers_.size();
  }

  ForwardWrapper *GetForwardWrapper(int forward_index) {
    return forward_wrappers_[forward_index];
  }

  /**
   * generate related pub topic for for this module's forward function
   * @param forward_idx forward index
   * @param input_slot input slot
   * @retunr error code
   */
  int GenSubTopic(int forward_idx, int input_slot);

  /**
   * generate related sub topic for this module's output
   * @param output_slot this module output slot
   * @retunr error code
   */
  int GenPubTopic(int output_slot);

  const std::vector<std::pair<int, int>> &SubTopics();

  const std::vector<int> &PubTopics();

  virtual ~Module() { }

 protected:
  void RegisterForwardWrapper(ForwardWrapper *forward_wrapper,
                              size_t forward_index) {
    if (forward_wrappers_.size() < forward_index + 1) {
      forward_wrappers_.resize(forward_index + 1);
    }
    forward_wrappers_[forward_index] = forward_wrapper;
  }

  void RegisterForwardCheck(CheckForward checkforward) {
      checkforward_ = checkforward;
  }

 public:
  std::map<std::string, fnConfigUpdate> configUpdateFns_;
  bool inited_;
  static CheckForward checkforward_;

 private:
  std::string class_name_;

  std::string instance_name_;

  Config config_;

  std::vector<ForwardWrapper *> forward_wrappers_;

  std::vector<std::pair<int, int>> sub_topics;
  std::vector<int> pub_topics;
};

class HOBOT_EXPORT InputModule: public Module {
 public:
  explicit InputModule(std::string instance_name = "")
      : Module(instance_name, "Input") {
  }

  FORWARD_DECLARE(InputModule, 0);

  int Init(RunContext * /*context*/) override {
    this->GenSubTopic(0, 0);
    return 0;
  }

  void Reset() override {
  }
};

class GroupModuleImp;
class HOBOT_EXPORT GroupModule : public Module {
 public:
    explicit GroupModule(std::string instance_name = "")
        : Module(instance_name, "GroupModule") {
    }
    virtual ~GroupModule();

    FORWARD_DECLARE(GroupModule, 0);

    /**
    * bind the group module with all child modules.
    * group must add the forward 1 of itself into the outputs modules, otherwise it can't work.
    * @param outputs collection of Modules whose output would be gathered by RunObserver, .
    */
    bool SetModules(const std::vector<Module*>& modules,
        int forward_index, Workflow *workflow,
        std::vector<std::pair<Module *, int>> *outputs);

    int Init(RunContext *context) override;
    void Reset() override;

 protected:
    GroupModuleImp * imp_ = nullptr;
};

/**
 * context object associated to one Workflow::Run() invocation.
 */
class HOBOT_EXPORT RunContext {
 public:
  virtual Engine *GetEngine() = 0;
  virtual Workflow *GetWorkflow() = 0;
  virtual Message *GetGlobalMessage() = 0;

  /**
   * Init reasonable modules in order
   * @param module return the moudle which init failed
   * @param init_order
   * @param verbose verbose mode
   */
  virtual int Init(Module **module = nullptr,
                   std::vector<Module*> *init_order = nullptr,
                   bool verbose = false) = 0;

  virtual ~RunContext() { }
};

/**
 * Envelop base class for all data objects input/output of Module.
 */
class HOBOT_EXPORT Message {
 public:
  virtual ~Message();
};

enum {
  ReqAll = -1
};

class HOBOT_EXPORT Expression {
 public:
  virtual ~Expression() { }

  /**
   * returns an AND expression: `a && b`.
   * AND expression's FetchAndEvaluate() fetches `a` and `b`
   * IFF `a` and `b` are both true; otherwise `data` is left unchanged.
   * @param a expression
   * @param b expression
   * @return AND expression
   */
  static Expression *And(Expression *a, Expression *b);

  /**
   * returns an OR expression: `a || b`.
   * OR expression's FetchAndEvaluate() fetches `a` if `a` evaluates to true;
   * if not, fetches `b` if `b` evaluates to true;
   * otherwise `data` is left unchanged.
   * @param a
   * @param b
   * @return OR expression
   */
  static Expression *Or(Expression *a, Expression *b);

  /**
   * returns an Require expression.
   * Require expression FetchAndEvaluate() success
   * if the data[index] has at least `count` messages.
   * if `count` == hobot::ReqAll, then this `Require` always evaluates to true,
   * and all messages in the `index`'th of `data` are fetched.
   * @param index index of data
   * @param count count of messages in data[index]
   * @param pass_if_no_link
            if input_slot[index] is NULL or index is out of range
            evaluate returns this value
   * @return Require expression
   */
  static Expression *Require(int index, int count = 1,
                             bool pass_if_no_link = true);

  /**
   * return an expression: `this && b`
   * @param b another expression
   * @return the AND expression
   */
  virtual Expression *And(Expression *b) = 0;

  /**
   * return an expression: `this || b`
   * @param b another expression
   * @return the OR expression
   */
  virtual Expression *Or(Expression *b) = 0;

  /**
   * Evaluate this expression;
   * fetch data into buffer if expression evaluates to true
   * @param data data to evaluate & fetch from
   * @param buffer empty vector to fetch into
   * @return true if evaluate passes and fetch success
   */
  virtual bool EvaluateAndFetch
      (const MessageLists &data, const MessageLists &buffer) = 0;

  /**
   * Evaluate this expression only
   * @param data data to evaluate
   * @return true if evaluate passes
   */
  virtual bool Evaluate(const MessageLists &data) = 0;

  /**
   * Fetch data into buffer this expression only
   * @param data data to fetch from
   * @param buffer empty vector to fetch into
   * @return true if fetch success
   */
  virtual bool Fetch(const MessageLists &data,
                     const MessageLists &buffer) = 0;


  /**
  * Evaluate this expression;
  * fetch data into buffer if expression evaluates to true
  * using links to skip input slots that not linked
  * @param data data to evaluate & fetch from
  * @param buffer empty vector to fetch into
  * @return true if evaluate passes and fetch success
  */
  virtual bool EvaluateAndFetch(const std::vector<Link *> &links,
                                const MessageLists &data,
                                const MessageLists &buffer) = 0;

  /**
  * Evaluate this expression only
  * using links to skip input slots that not linked
  * @param data data to evaluate
  * @return true if evaluate passes
  */
  virtual bool Evaluate(const std::vector<Link *> &links,
                        const MessageLists &data) = 0;

  /**
  * Fetch data into buffer this expression only
  * using links to skip input slots that not linked
  * @param data data to fetch from
  * @param buffer empty vector to fetch into
  * @return true if fetch success
  */
  virtual bool Fetch(const std::vector<Link *> &links,
                     const MessageLists &data,
                     const MessageLists &buffer) = 0;
};

// message handler
class MsgHandler {
 public:
  /**
   * subscription interface
   * @param topic topic
   * @param message message related to the topic
   * @return bool value
   */
  virtual bool Handle(const std::string &topic, spMessage message) = 0;
};

// pub-sub client
class HOBOT_EXPORT Client {
 public:
  /**
   * subscription interface
   * @param topic topic
   * @param msg_handler topic related callback
   * @param msg_handler topic related handler shared pointer
   * @return error code
   */
  virtual int Sub(const std::string &topic, const MsgCallback &msg_callback,
                  spMsgHandler handler = nullptr) = 0;
  /**
   * unsubscribe the topic
   * @param topic topic
   * @return error code
   */
  virtual int UnSub(const std::string &topic) = 0;

  /**
   * publish interface
   * @param topic topic
   * @param topic related message
   * @return error code
   */
  virtual int Pub(const std::string &topic, spMessage message) = 0;

  virtual ~Client() {}
};

enum class BrokerType {
  DEFAULT,
};
// pub-sub broker
class HOBOT_EXPORT Broker {
 public:
  /**
   * Create a broker which can not guarantee thread safety when dynamic sub
   * occurs. This broker is enough in most cases.
   * @param type indicate which type of broker, now only DEFAULT type available
   * @return broker shared pointer
   */
  static spBroker NewInstance(BrokerType type = BrokerType::DEFAULT);

  /**
   * create a broker, guarantee thread safety when dynamic sub occurs.
   * @param type indicate which type of broker, now only DEFAULT type available
   * thread_idx within broker
   * @return broker shared pointer
   */
  static spBroker NewInstanceThreadSafe(BrokerType type = BrokerType::DEFAULT);

  /**
   * fetch a pub-sub client connected to the broker
   * @param thread_idx client's sub callback will be run in the thread of
   * thread_idx within broker
   * @return client shared pointer
   */
  virtual spClient NewClient(uint32_t thread_idx = 0) = 0;

  /**
   * set max size of buffer used to keep unconsumed messages for the topic
   * @param topic topic
   * @param size the max size for message buffer for the topic
   * @return error code
   */
  virtual int SetBufferSize(const std::string &topic, int size = INT32_MAX) = 0;

  /**
   * get the max size of buffer used to keep unconsumed messages for the topic
   * @param topic topic
   * @return error code
   */
  virtual int BufferSize(const std::string &topic) = 0;

  /**
   * fetch the BrokerStatus object of this broker
   * @param  none
   * @return broker status shared pointer
   */
  virtual spBrokerStatus BrokerStatus() = 0;

  virtual ~Broker() {}
};

// pub-sub broker's status
class HOBOT_EXPORT BrokerStatus {
 public:
  /**
   * get the buffer size currently in use for the topic
   * @param  topic topic
   * @return  buffer size currently in us
   */
  virtual int BufferSizeInUse(const std::string &topic) = 0;

  virtual ~BrokerStatus() {}
};

}  // namespace hobot

#endif  // HOBOT_HOBOT_H_
