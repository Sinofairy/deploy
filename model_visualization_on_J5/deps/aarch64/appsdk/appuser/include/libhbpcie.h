/*************************************************************************
 *                     COPYRIGHT NOTICE
 *            Copyright 2016-2022 Horizon Robotics, Inc.
 *                   All rights reserved.
 *************************************************************************/
#ifndef LIBHBPCIE_H_
#define LIBHBPCIE_H_
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "pcie-hobot.h"


#define MAP_PCI_MAX_SIZE 0xffffffffu
#define MAP_PCI_ADDR_SIZE_ALIGN 0x1000u


#define HB_PCIE_ERR_NONE						0  /* no error */
#define HB_PCIE_ERR_INTERRUPTTED				-1 /* operation was interruptted */
#define HB_PCIE_ERR_INVAILD_CHAN_ID				-2 /* invalid channel id */
#define HB_PCIE_ERR_NO_FREE_HANDLE				-3 /* no more free handle */
#define HB_PCIE_ERR_OUT_OF_MEM					-4 /* out of memory */
#define HB_PCIE_ERR_OPEN_DEV					-5 /* open PCIe device fail */
#define HB_PCIE_ERR_VERSION_MISMATCH			-6 /* library and driver version mismatch */
#define HB_PCIE_ERR_CHAN_ALREADY_ALLOC			-7 /* channel already in use */
#define HB_PCIE_ERR_MAP_BAR						-8 /* map BAR fail */
#define HB_PCIE_ERR_INVALID_HANDLE				-9 /* invalid handle */
#define HB_PCIE_ERR_INVALID_MODE				-10 /* invaild DMA mode */
#define HB_PCIE_ERR_INVALID_ELEMENTS			-11 /* invaild DMA elements */
#define HB_PCIE_ERR_INVALID_WEIGHT				-12 /* invaild DMA weight */
#define HB_PCIE_ERR_STL_NOT_READY				-13 /* STL(FUSA) not ready */
#define HB_PCIE_ERR_LINK_DOWN					-14 /* PCIe link down */
#define HB_PCIE_ERR_DMA_XFER_TIMEOUT			-15 /* DMA transfer timeout */
#define HB_PCIE_ERR_DMA_XFER_ABORTED			-16 /* DMA transfer aborted */
#define HB_PCIE_ERR_DMA_HANDSHAKE				-17 /* DMA transfer done but wait remote handshake timeout */
#define HB_PCIE_ERR_RECEIVER_EXIST				-18 /* receiver already installed */
#define HB_PCIE_ERR_RECEIVER_NOT_EXIST			-19 /* receiver not installed yet */
#define HB_PCIE_ERR_THREAD_CREATE				-20 /* start receiver thread fail */
#define HB_PCIE_ERR_THREAD_STOP					-21 /* stop receiver thread fail */
#define HB_PCIE_ERR_INVALID_BAR_ID				-22 /* invalid BAR id for read/write */
#define HB_PCIE_ERR_INVAILD_BAR_OFFSET			-23 /* invalid BAR offset for read/write */
#define HB_PCIE_ERR_UNSUPPORT_OPERATION			-24 /* un-support opeartion */
#define HB_PCIE_ERR_RAISE_IRQ					-25 /* raise IRQ fail */
#define HB_PCIE_ERR_INVALID_INT_ID				-26 /* invalid interrupt ID */
#define HB_PCIE_ERR_HANDLER_EXIST				-27 /* interrupt handler already installed */
#define HB_PCIE_ERR_HANDLER_NOT_OWN				-28 /* interrupt handler not owned by requester */
#define HB_PCIE_ERR_HANDLER_NOT_EXIST			-29 /* interrupt handler not installed yet */
#define HB_PCIE_ERR_INVALID_PARAM				-30 /* invalid param */
#define HB_PCIE_ERR_NETLINK_CREATE				-31 /* create netlink socket fail */
#define HB_PCIE_ERR_NETLINK_BIND				-32 /* bind netlink socket fail */
#define HB_PCIE_ERR_NETLINK_SETOPT				-33 /* config netlink socket fail */
#define HB_PCIE_ERR_NO_PCI_MEM					-34 /* no free PCI space available */
#define HB_PCIE_ERR_MAP_PCI_NO_OUTBOUND			-35 /* no free outbound windows available */
#define HB_PCIE_ERR_UNMAP_PCI_NOT_MAPPED		-37 /* address of PCI memory not mapped yet */
#define HB_PCIE_ERR_UNMAP_PCI_NOT_OWN			-38 /* the mapped memory is not owned by requester */
#define HB_PCIE_ERR_DMA_REQUESTED				-39 /* DMA channel already requested by this handle */
#define HB_PCIE_ERR_DMA_NOT_REQUESTED			-40 /* DMA channel not requested by this handle */
#define HB_PCIE_ERR_MAP_SIZE_INVALID			-41 /* invaild map size */
#define HB_PCIE_ERR_MAP_ADDR_INVALID			-42 /* invaild map addr */
#define HB_PCIE_ERR_WRITE_RESERVED_BAR			-43 /* can not write to reserved BAR region */
#define HB_PCIE_ERR_NOT_INIT_BY_RC				-44 /* device not init by RC */
#define HB_PCIE_ERR_IN_SUSPEND					-45 /* remote device in suspend status */
#define HB_PCIE_ERR_UNKNOWN						-46 /* unkown error */
#define HB_PCIE_ERR_INVAILD_IOCTL_CMD			-47 /* invalid ioctl command */
#define HB_PCIE_ERR_MEM_COPY					-48 /* copy from/to kernel fail */
#define HB_PCIE_ERR_REMOTE_RECV_NOT_EXIST		-49 /* remote receiver not installed */
#define HB_PCIE_ERR_RECEIVER_NOT_OWN			-50 /* receiver is not owned by requester */
#define HB_PCIE_ERR_PTHREAD_MUTEX				-51 /* pthread mutex lock fail */


typedef enum {
	mode_none = 0,
	mode_wr_only,
	mode_rd_only,
	mode_wr_rd,
} dma_mode;


/**
 * hb_pcie_init() - Request and initialize device resource and return the handle id.
 * @bus_id: the bus id of device
 * @dev_id: the device id of device
 *
 * on EP side, both bus_id and dev_id fix to "0"
 * on RC side, EP bus id and device id can be obtained by lspci:
 *   root@j5dvb-hynix4G:~# lspci | grep 16c3:1103
 *   04:00.0 Class ff00: Device 16c3:1103
 *   05:00.0 Class ff00: Device 16c3:1103
 *   06:00.0 Class ff00: Device 16c3:1103
 * there 3 EP devices are detected:
 *   dev1: "04" is the bus id and "00" is the device id
 *   dev2: "05" is the bus id and "00" is the device id
 *   dev3: "06" is the bus id and "00" is the device id
 *
 * Return: >=0 success, or error code
 */
int32_t hb_pcie_init(int32_t bus_id, int32_t dev_id);


/**
 * hb_pcie_deinit - Free requested resource.
 * @handle_id: handle returned from hb_pcie_init
 *
 * Return: 0 success, or error code
 */
int32_t hb_pcie_deinit(int32_t handle_id);

/**
 * hb_pcie_request_dma() - Request DMA channel resource.
 * @handle_id: handle returned from hb_pcie_init
 * @channel: requested DMA channel ID, available range: 0 ~ 3
 * @mode: DMA operation mode, available value:
 *    mode_none: do not request any channel's read or write resource,
 *               typical used for install receiver callback
 *    mode_wr_only: only request channel's write resource
 *    mode_rd_only: only request channel's read resource
 *    mode_wr_rd: request channle's write and read resource
 * 
 * Each handle can request only one DMA channel
 *
 * Return: 0 success, or error code
 */
int32_t hb_pcie_request_dma(int32_t handle_id, int32_t channel_id, dma_mode mode);

/**
 * hb_pcie_release_dma() - Release DMA channel resource.
 * @handle_id: handle returned from hb_pcie_init
 *
 * Must release dma after recevier uninstalled
 * 
 * Return: 0 success, or error code
 */
int32_t hb_pcie_release_dma(int32_t handle_id);

/**
 * hb_pcie_dma_write() - Write data to remote memory via DMA.
 * @handle_id: handle returned from hb_pcie_init
 * @eles: the elements for DMA transfer.
 *        struct element {
 *            uint32_t size; transfer size
 *            uint64_t dest; destitation physics address
 *            uint64_t src; source physics address
 *        }
 * @ele_nr: the number of elements, available range: 1 ~ 8
 * @weight: DMA controller uses a weighted round robin (WRR) arbitration scheme to select the
 *          next channel to be serviced, the bandwidth for all channels is shared in time and is 
 *          controlled by a weighted round-robin arbiter.
 *          available range: 1 ~ 31
 * 
 * Return: 0 success, or error code
 */
int32_t hb_pcie_dma_write(int32_t handle_id, const struct element *eles, int32_t ele_nr, int32_t weight);


/**
 * hb_pcie_dma_write_strict() - Write data to remote memory via DMA if remote receiver installed.
 * @handle_id: handle returned from hb_pcie_init
 * @eles: the elements for DMA transfer.
 *        struct element {
 *            uint32_t size; transfer size
 *            uint64_t dest; destitation physics address
 *            uint64_t src; source physics address
 *        }
 * @ele_nr: the number of elements, available range: 1 ~ 8
 * @weight: DMA controller uses a weighted round robin (WRR) arbitration scheme to select the
 *          next channel to be serviced, the bandwidth for all channels is shared in time and is 
 *          controlled by a weighted round-robin arbiter.
 *          available range: 1 ~ 31
 *
 * If remote receiver not installed, an error will be return. 
 * 
 * Return: 0 success, or error code
 */
int32_t hb_pcie_dma_write_strict(int32_t handle_id, const struct element *eles, int32_t ele_nr, int32_t weight);


/**
 * hb_pcie_dma_read() - Read data from remote memory via DMA.
 * @handle_id: handle returned from hb_pcie_init
 * @eles: the elements for DMA transfer.
 *        struct element {
 *            uint32_t size; transfer size
 *            uint64_t dest; destitation physics address
 *            uint64_t src; source physics address
 *        }
 * @ele_nr: the number of elements, available range: 1 ~ 8
 * @weight: DMA controller uses a weighted round robin (WRR) arbitration scheme to select the
 *          next channel to be serviced, The bandwidth for all channels is shared in time and is 
 *          controlled by a weighted round-robin arbiter.
 *          available weight: 1 ~ 31
 * 
 * Return: 0 success, or error code
 */
int32_t hb_pcie_dma_read(int32_t handle_id, const struct element *eles, int32_t ele_nr, int32_t weight);


/**
 * hb_pcie_install_receiver() - Install DMA data receive callback function.
 * @handle_id: handle returned from hb_pcie_init
 * @receiver: the callback function
 * @data: user pointer can be used in callback function
 *
 * once the DMA data come from remote by hb_pcie_dma_write/hb_pcie_dma_write_strict, installed
 * receive function will be callback, conflict with the mode_rd_only and mode_wr_rd mode.
 * 
 * while remote write data repeatedly, the receiver function will be invoked sequentially, so 
 * it's recommend to handle the data as soon as possible, to avoid data dropped casued by the 
 * receiver buffer full.
 * 
 * Do not call the hb_pcie_deinit or hb_pcie_uninstall_receiver in receiver function.
 *
 * Return: 0 success, or error code
 */
int32_t hb_pcie_install_receiver(int32_t handle_id, void (*receiver)(int32_t channel_id,
										const struct element *eles, int32_t ele_nr, void *data),
										void *data);


/**
 * hb_pcie_uninstall_receiver() - Un-install DMA data receive callback function.
 * @handle_id: handle returned from hb_pcie_init
 *
 * Return: 0 success, or error code
 */
int32_t hb_pcie_uninstall_receiver(int32_t handle_id);


/**
 * hb_pcie_remote_recevier_installed() - check whether the channel's remote recevier installed.
 * @handle_id: handle returned from hb_pcie_init
 * @channel_id: channel id
 * 
 * Return: 0 success, or error code
 */
int32_t hb_pcie_remote_recevier_installed(int32_t handle_id, int32_t channel_id);


/**
 * hb_pcie_bar_read() - Read data from BAR.
 * @handle_id: handle returned from hb_pcie_init
 * @bar_id: BAR id, available value: 2, 4, 5
 * @offset: the offset of BAR
 * @buf: store the read data
 * @len: read length
 * 
 * The memory from offset 0 to RESERVED_END_OFFSET in BAR2 is reserved by system.
 *
 * Return: 0 success, or error code
 */
int32_t hb_pcie_bar_read(int32_t handle_id, int32_t bar_id, uint64_t offset, void *buf, uint64_t len);


/**
 * hb_pcie_bar_write() - Write data to BAR.
 * @handle_id: handle returned from hb_pcie_init
 * @bar_id: BAR id, available value: 2, 4, 5
 * @offset: the offset of BAR
 * @buf: data to be wroten
 * @len: write length
 *
 * The memory from offset 0 to RESERVED_END_OFFSET in BAR2 is reserved by system.
 *
 * Return: 0 success, or error code
 */
int32_t hb_pcie_bar_write(int32_t handle_id, int32_t bar_id, uint64_t offset, const void *buf, uint64_t len);


/**
 * hb_pcie_get_bar_info() - Get BAR physics address and size.
 * @handle_id: handle returned from hb_pcie_init
 * @bar_id: BAR id, available value: 2, 4, 5
 * @phys_addr: return the BAR physics address, on RC side: PCI address, on EP side: DDR address.
 * @size: BAR size
 * 
 * Return: 0 success, or error code
 */
int32_t hb_pcie_get_bar_info(int32_t handle_id, int32_t bar_id, uint64_t *phys_addr, uint64_t *size);

/**
 * hb_pcie_get_bar_phys() - Get BAR physics address and size.
 * @handle_id: handle returned from hb_pcie_init
 * @bar_id: BAR id, available value: 2, 4, 5
 * @cpu_phys: return the BAR DDR physics address in EP
 * @pci_phys: return the BAR PCI physics address
 * @size: BAR size
 * 
 * Return: 0 success, or error code
 */
int32_t hb_pcie_get_bar_phys(int32_t handle_id, int32_t bar_id,
			uint64_t *cpu_phys, uint64_t *pci_phys, uint64_t *size);

/**
 * hb_pcie_map_rc_mem() - Map RC memory to EP local address, only availabe on EP side.
 * @handle_id: handle returned from hb_pcie_init
 * @rc_phys: the physics address of RC memory to be mapped, the address must be 4KB align
 * @mapped_phys: the mmaped EP local physics address
 * @size: map size, available range: 4KB ~ 4GB, the size must be 4KB align
 * 
 * Return: 0 success, or error code
 */
int32_t hb_pcie_map_rc_mem(int32_t handle_id, uint64_t rc_phys, uint64_t *mapped_phys, uint64_t size);

/**
 * hb_pcie_unmap_rc_mem() - Unmap RC memory in EP local address, only availabe on EP side.
 * @handle_id: handle returned from hb_pcie_init
 * @pci_phys: the mmaped EP local physics address
 * 
 * Return: 0 success, or error code
 */
int32_t hb_pcie_unmap_rc_mem(int32_t handle_id, uint64_t mapped_phys);

/**
 * hb_pcie_map_pci_addr() - Map PCI address to EP local address, only availabe on EP side.
 * @handle_id: handle returned from hb_pcie_init
 * @pci_phys: the physics address of PCI address to be mapped, the address must be 4KB align
 * @mapped_phys: mmaped EP local physics address
 * @size: map size, available range: 4KB ~ 4GB, the size must be 4KB align
 * 
 * the RC DDR memory and the PCI address has the same effect.
 * 
 * Return: 0 success, or error code
 */
int32_t hb_pcie_map_pci_addr(int32_t handle_id, uint64_t pci_phys, uint64_t *mapped_phys, uint64_t size);

/**
 * hb_pcie_unmap_pci_addr() - Unmap PCI address in EP local address, only availabe on EP side.
 * @handle_id: handle returned from hb_pcie_init
 * @mapped_phys: the mmaped EP local physics address
 * 
 * Return: 0 success, or error code
 */
int32_t hb_pcie_unmap_pci_addr(int32_t handle_id, uint64_t mapped_phys);

/**
 * hb_pcie_get_max_interrupt_num() - Get the max number of interrupts.
 * @handle_id: handle returned from hb_pcie_init
 * 
 * Return: >0 the number of support interrupts, or error code
 */
int32_t hb_pcie_get_max_interrupt_num(int32_t handle_id);


/**
 * hb_pcie_trigger_interrupt() - Send interrupt to RC, only availabe on EP side.
 * @handle_id: handle returned from hb_pcie_init
 * @id: interrupt ID, start from 1, the max value can be get from hb_pcie_get_max_interrupt_num()
 * 
 * Return: 0 success, or error code
 */
int32_t hb_pcie_trigger_interrupt(int32_t handle_id, int32_t id);


/**
 * hb_pcie_install_interrupt_handler() - Install the interrupt handler, only availabe on RC side.
 * @handle_id: handle returned from hb_pcie_init
 * @id: interrupt ID
 * @handler: callback function once get interrupt from EP
 * @data: user pointer can be used in callback function
 *
 * Each interrupt ID only support install one handler.
 * Do not call the hb_pcie_deinit or hb_pcie_uninstall_interrupt_handler in handler function.
 * 
 * Return: 0 success, or error code
 */
int32_t hb_pcie_install_interrupt_handler(int32_t handle_id, int32_t id, void (*handler)(int32_t id, void *data),
											void *data);


/**
 * hb_pcie_uninstall_interrupt_handler() - Un-install the interrupt handler, only availabe on RC side.
 * @handle_id: handle returned from hb_pcie_init
 * @int_id: interrupt ID
 * 
 * Return: 0 success, or error code
 */
int32_t hb_pcie_uninstall_interrupt_handler(int32_t handle_id, int32_t id);

/**
 * hb_pcie_get_bus_info() - Get bus/device/function id, only availabe on EP side.
 * @handle_id: handle returned from hb_pcie_init
 * @bus_id: bus id of EP device
 * @dev_id: device id of EP device
 * @fun_id: function id of EP device
 *
 * Return: 0 success, or error code
 */
int32_t hb_pcie_get_bus_info(int32_t handle_id, uint32_t *bus_id, uint32_t *dev_id, uint32_t *fun_id);

#ifdef HOBOT_PCIE_CUSTOM_MEM
/* debug purpose for x86_64 PC as RC */
int32_t hb_pcie_alloc_mem(int32_t handle_id, int32_t size, uint64_t *phys);
int32_t hb_pcie_free_mem(int32_t handle_id, uint64_t phys);
#endif

#ifdef __cplusplus
}
#endif
#endif // LIBHBPCIE_H_
