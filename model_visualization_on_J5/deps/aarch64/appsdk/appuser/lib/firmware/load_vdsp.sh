#! /bin/sh
### START VDSP
# Provides:             vdsp
# Required-Start:
# Required-Stop:
# Default-Start:
# Default-Stop:
# Short-Description:    load vdsp image
### END VDSP
cd /system/lib/firmware
echo "start load vdsp image"
echo vdsp0 > /sys/class/remoteproc/remoteproc1/firmware
echo vdsp1 > /sys/class/remoteproc/remoteproc2/firmware
echo start > /sys/class/remoteproc/remoteproc1/state
echo start > /sys/class/remoteproc/remoteproc2/state
cd -
exit 0
