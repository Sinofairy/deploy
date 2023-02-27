#! /bin/sh
### BEGIN INIT INFO
# Provides:             echo_virtio
# Required-Start:
# Required-Stop:
# Default-Start:
# Default-Stop:
# Short-Description:    register_virtio
### END INIT INFO
cd /system/lib/firmware
echo "register virtio to mcore"
echo register_virtio >  /sys/class/remoteproc/remoteproc0/state
sleep 6
devmem 0x2001bf28 32 0xdeadbeef
cd -
exit 0

