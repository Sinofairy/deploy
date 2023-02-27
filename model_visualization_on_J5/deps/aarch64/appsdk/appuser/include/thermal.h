#ifndef THERMAL_H
#define THERMAL_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	THROTTLING_NONE,
	THROTTLING_LIGHT,
	THROTTLING_MODERATE,
	THROTTLING_SEVERE,
	THROTTLING_CRITICAL,
	THROTTLING_EMERGENCY,
	THROTTLING_SHUTDOWN,
} ThermalStatusType;

typedef enum {
	TYPE_UNKNOWN,
	TYPE_CPU,
	TYPE_GPU,
	TYPE_BATTERY,
	TYPE_SKIN,
	TYPE_USB_PORT,
	TYPE_POWER_AMPLIFIER,
	TYPE_BCL_VOLTAGE,
	TYPE_BCL_CURRENT,
	TYPE_BCL_PERCENTAGE,
	TYPE_BPU,
	TYPE_MAX,
} ThermalDeviceType;

typedef struct {
	unsigned long temp;
} Temperature;

/**
 * ThermalEventListener - Callback to notify application about device
 * temperature event.
 * @type: type id for thermal device. get it frome hobot_query_threamal_type.
 * @is_throttling: The system performance have been limited due to
 * high temperature.
 * @temp: temperature now.
 *
 * This callback will be call when system performance was limited due
 * to high temperature.
 * This callback will execute in a background thread, this thread will
 * be created automatically when you call
 * @hobot_register_thermal_event_listener or @hobot_register_thermal_event_listener_with_type
 * refs to http://androidxref.com/9.0.0_r3/xref/frameworks/native/services/thermalservice/aidl/android/os/IThermalEventListener.aidl
 *
 * Return: void
 */
typedef void (*ThermalEventListener)(int type, int is_throttling, Temperature temp);

/**
 * ThermalStatusListener - Callback to notify application about thermal
 * status event.
 * @status: The system thermal status, refs to @ThermalStatusType
 *
 * This callback will be call when system thermal status is changed.
 * This callback will execute in a background thread, this thread will
 * be created automatically when you call
 * @hobot_register_thermal_status_listener
 * refs to https://android.googlesource.com/platform/frameworks/base/+/master/core/java/android/os/Temperature.java
 *
 * Return: void
 */
typedef void (*ThermalStatusListener)(ThermalStatusType status);

/**
 * hobot_query_threamal_type - get system thermal types
 * @type: An array to save thermal_types in.
 * @len: length of type.
 *
 * This function will enumerate thermal device automatically, and
 * return a thermal_type list.
 *
 * Return: length of thermal_type in system. < 0: in case of error.
 */
int hb_query_thermal_type(int type[], int len);

/**
 * hobot_register_thermal_event_listener - register a
 * ThermalEventListener.
 * @listener: listener to be registered
 *
 * This function will create a thermal therad automatically, and
 * execute callback function in the created thread.
 *
 * Return: = 0 on success, < 0: in case of error.
 */
int hb_register_thermal_event_listener(ThermalEventListener listener);

/**
 * hb_register_thermal_event_listener_with_type - register a
 * ThermalEventListener.
 * @type: type id from @hb_query_thermal_type
 * @listener: listener to be registered
 *
 * This function will create a thermal therad automatically, and
 * execute callback function in the created thread.
 *
 * Return: = 0 on success, < 0: in case of error.
 */
int hb_register_thermal_event_listener_with_type(int type, ThermalEventListener listener);

int hb_unregister_thermal_event_listener(ThermalEventListener listener);

/**
 * hobot_register_thermal_status_listener - register a
 * ThermalStatusListener.
 * @listener: listener to be registered
 *
 * This function will create a thermal therad automatically, and
 * execute callback function in the created thread.
 *
 * Return: = 0 on success, < 0 on errors.
 */
int hb_register_thermal_status_listener(ThermalStatusListener listener);

/**
 * hb_unregister_thermal_status_listener - register a
 * ThermalStatusListener.
 * @listener: listener to be unregistered
 *
 * This function will create a thermal therad automatically, and
 * execute callback function in the created thread.
 *
 * Return: 0.
 */
int hb_unregister_thermal_status_listener(ThermalStatusListener listener);

#ifdef __cplusplus
}
#endif

#endif
