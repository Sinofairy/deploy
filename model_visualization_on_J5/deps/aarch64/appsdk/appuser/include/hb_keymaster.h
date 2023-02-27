/*
 * Horizon Robotics
 *
 *  Copyright (C) 2020 Horizon Robotics Inc.
 *  All rights reserved.
 *  Author:
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
#include <stdint.h>

typedef int32_t key_serial_t;
typedef uint32_t key_perm_t;

#define HBKM_READ_OP_KEY			0xaa
#define HBKM_READ_OP_PUBKEY			0xbb

#define HBKM_SUPPORTS_ENCRYPT		0x01
#define HBKM_SUPPORTS_DECRYPT		0x02
#define HBKM_SUPPORTS_SIGN		0x04
#define HBKM_SUPPORTS_VERIFY		0x08

#define HBKM_KEY_TYPE_AES			0xA0000010
#define HBKM_KEY_TYPE_DES			0xA0000011
#define HBKM_KEY_TYPE_DES3			0xA0000013
#define HBKM_KEY_TYPE_HMAC_SHA1		0xA0000002
#define HBKM_KEY_TYPE_HMAC_SHA256	0xA0000004
#define HBKM_KEY_TYPE_HMAC_SHA512	0xA0000006
#define HBKM_KEY_TYPE_RSA_PUBKEY	0xA0000030
#define HBKM_KEY_TYPE_RSA_KEYPAIR	0xA1000030
#define HBKM_KEY_TYPE_ECDSA_PUBKEY	0xA0000041
#define HBKM_KEY_TYPE_ECDSA_KEYPAIR	0xA1000041

#define HBKM_KEY_POS_VIEW		0x01000000	/* possessor can view a key's attributes */
#define HBKM_KEY_POS_READ		0x02000000	/* possessor can read key payload / view keyring */
#define HBKM_KEY_POS_WRITE		0x04000000	/* possessor can update key payload / add link to keyring */
#define HBKM_KEY_POS_SEARCH		0x08000000	/* possessor can find a key in search / search a keyring */
#define HBKM_KEY_POS_LINK		0x10000000	/* possessor can create a link to a key/keyring */
#define HBKM_KEY_POS_SETATTR	0x20000000	/* possessor can set key attributes */
#define HBKM_KEY_POS_ALL		0x3f000000

#define HBKM_KEY_USR_VIEW		0x00010000	/* user permissions... */
#define HBKM_KEY_USR_READ		0x00020000
#define HBKM_KEY_USR_WRITE		0x00040000
#define HBKM_KEY_USR_SEARCH		0x00080000
#define HBKM_KEY_USR_LINK		0x00100000
#define HBKM_KEY_USR_SETATTR	0x00200000
#define HBKM_KEY_USR_ALL		0x003f0000

#define HBKM_KEY_GRP_VIEW		0x00000100	/* group permissions... */
#define HBKM_KEY_GRP_READ		0x00000200
#define HBKM_KEY_GRP_WRITE		0x00000400
#define HBKM_KEY_GRP_SEARCH		0x00000800
#define HBKM_KEY_GRP_LINK		0x00001000
#define HBKM_KEY_GRP_SETATTR	0x00002000
#define HBKM_KEY_GRP_ALL		0x00003f00

#define HBKM_KEY_OTH_VIEW		0x00000001	/* third party permissions... */
#define HBKM_KEY_OTH_READ		0x00000002
#define HBKM_KEY_OTH_WRITE		0x00000004
#define HBKM_KEY_OTH_SEARCH		0x00000008
#define HBKM_KEY_OTH_LINK		0x00000010
#define HBKM_KEY_OTH_SETATTR	0x00000020
#define HBKM_KEY_OTH_ALL		0x0000003f

#define HBKM_KEY_TYPE_SECURE_KEY "secure-key"
#define SECURE_KEY_NEW "new $len keytype=$type"

#define  HBKM_OK				0	/* No error */
#define  HBKM_ERR_GENERIC		-1	/* General error */
#define  HBKM_ERR_ACCES			-2	/* Permission denied */
#define  HBKM_ERR_BADMSG		-3	/* Not a data message */
#define  HBKM_ERR_DQUOT			-4	/* Quota exceeded */
#define  HBKM_ERR_FAULT			-5	/* Operation failed ,please check input param*/
#define  HBKM_ERR_INVAL			-6	/* Invalid argument */
#define  HBKM_ERR_KEYEXPIRED	-7	/* Key has expired */
#define  HBKM_ERR_KEYREJECTED	-8	/* Key was rejected by service */
#define  HBKM_ERR_KEYREVOKED	-9	/* Key has been revoked */
#define  HBKM_ERR_NOKEY			-10	/* Required key or keying not available */
#define  HBKM_ERR_NOMEM			-11	/* Out of memory */
#define  HBKM_ERR_NOFACILITY	-12	/* Some facility needed to complete the requested operation is not available. */
#define  HBKM_ERR_NOTKEYRING	-13	/* One of the keyrings is a valid key that isn't a keyring */
#define  HBKM_ERR_OPNOTSUPP		-14	/* The key type does not support the operation */
#define  HBKM_ERR_OVERFLOW		-15	/* Value too large for defined data type */
#define  HBKM_ERR_PERIOD		-16	/* Type started with period(.) or type was keyring and description started with period.*/
#define  HBKM_ERR_BUFSHORT		-17	/* Input buffer was too short */
#define  HBKM_ERR_KEYTYPE		-18 /* Key type was error when add key */
#define  HBKM_ERR_KEYNOLINK		-19 /* Key was not link to the keyring*/

struct hbkm_raw_key_header{
	unsigned int magic;
	unsigned int type;
	unsigned int format;
	unsigned int keylen;
	unsigned int offset;
	unsigned int size;
};

struct hbkm_pkey_query {
	unsigned int	supported_ops;	/* Which ops are supported */
	unsigned int	key_size;	/* Size of the key in bits */
	unsigned short	max_data_size;	/* Maximum size of raw data to sign in bytes */
	unsigned short	max_sig_size;	/* Maximum size of signature in bytes */
	unsigned short	max_enc_size;	/* Maximum size of encrypted blob in bytes */
	unsigned short	max_dec_size;	/* Maximum size of decrypted blob in bytes */
	unsigned int	__spare[10];
};

/*
	in: type: key type, now only support 'secure-key'
		description: key name, it can be defined by user. key name
		can't be the same of exsited keys
		payload: used to set option, it can be two format:
		(1) new [len] keytype=[type]
			eg: new 1024 keytype=A1000030
			len is the key legnth in bit, type is the key type
		(2) load [data]
			data is the real key payload
		keytype definition:
		(1)AES  A0000010
		(2)DES  A0000011
		(3)DES3  A0000013
		(4)HMAC_SHA1  A0000002
		(5)HMAC_SHA256  A0000004
		(6)HMAC_SHA512  A0000006
		(7)RSA KEYPAIR  A1000030
		(8)ECDSA KEYPAIR  A1000041
	out: void
*/
key_serial_t hbkm_add_key(const char *type,
				const char *description,
				const void *payload,
				size_t plen,
				key_serial_t ringid);

/*
	in: id: key serial id
	out: void
*/
long hbkm_invalidate(key_serial_t id);

/*
	in: id: key serial id
		payload: key payload data
		plen: payload length
	out: void
*/
long hbkm_update_key(key_serial_t id, const void
				*payload, size_t plen);

/*
	in: id: key serial id
	out: void
*/
long hbkm_revoke_key(key_serial_t id);

/*
	in: id: key serial id
		uid: the new uid of this key
		gid: the new gid of this key
	out: void
*/
long hbkm_chown(key_serial_t id, uid_t uid,
				gid_t gid);

/*
	in: id: key serial id
		perm: key permission, the permission is defined by a u32 value. it is divided to four
		group, each group include 8 bits.
		(1) bit0 -bit7: other permission
		(2) bit8 - bit15: group permission
		(3) bit16 - bit23: user permission
		(4) bit24 - bit31: possession permission
	out: void
*/
long hbkm_setperm(key_serial_t id, key_perm_t perm);

/*
	in: id: key serial id
			buffer: buffer used to return key describe information
			buflen: buffer length
		description information includes:
		(1) key id
		(2) key permission
		(3) key owner uid, gid
		(4) key type
		(5) key name
	out: void
*/
long hbkm_describe(key_serial_t id, char *buffer,
				size_t buflen);

/*
	in: ringid: keyring id
	out: void
*/
long hbkm_clear(key_serial_t ringid);

/*
	in: ringid: keyring id
		type: key type
		description: key description information
		destringid: dest keyring id, this parameter is option
	out: void
*/
long hbkm_search(key_serial_t ringid,
				const char *type,
				const char *description,
				key_serial_t destringid);

/*
	in:id: key serial id

	out: buffer: buffer to reture read key payload
		 buflen: buffer length
*/
long hbkm_read(key_serial_t id, char *buffer,
				size_t buflen);

/*
	in: id: key id
	out: buffer: buffer to reture read public key payload, it is pem format
		 buflen: buffer length
*/
long hbkm_pubkey_read(key_serial_t id, char *buffer,
				size_t buflen);

/*
	in: id: key serial id
		timeout: timeout value
	out: void
*/
long hbkm_set_timeout(key_serial_t id,
				unsigned timeout);

/*
	in: id: key serial id
	out: buffer: buffer used to return lsm information
		buflen: buffer length
*/
long hbkm_get_security(key_serial_t id,
				char *buffer, size_t buflen);

/*
	in: void
	out: void
	ret: key numbers which is been reaped
*/
long hbkm_reap(void);

/*
	in: id: key serial id
		ringid: keyring id
	out: void
*/
long hbkm_key_unlink(key_serial_t id, key_serial_t ringid);

/*
	in: key_id: key serial id
		info: retrive options, option format is 'key=value',
		option supported is 'enc=algo'.which used to config algorithm.
		algo been defined is below:
		(1)rsa-nopad-tee
		(2)rsa-v1_5-es-tee
		(3)rsa-v1_5-sha1-tee
		(3)rsa-v1_5-sha256-tee
		(4)sa-v1_5-sha512-tee
		(5)rsa-oaep-sha1-tee
		(6)rsa-oaep-sha256-tee
		(7)rsa-oaep-sha512-tee
		(8)rsa-pss-sha1-tee
		(9)rsa-pss-sha256-tee
		(10)rsa-pss-sha512-tee
		(11)ecdsa-p192-tee
		(12)ecdsa-p256-tee
		(13)ecdsa-p512-tee
	out: result: key atribute retrived from kernel.
		it includes below attribute:
		(1)operation supported by pkey, it may encrypt, decrypt, signature,
			verify, or some of them
		(2)key size
		(3)max input data length supported
		(4)max signature length supported
		(5)max encryption data length supported
		(6)max decryption data length supported
*/
long hbkm_pkey_query(key_serial_t key_id,
			const char *info,
			struct hbkm_pkey_query *result);

/*
	in: key_id: key serial id
		info: ref to hbkm_pkey_query() info param
		data: plain text
		data_len: plain text lenght
	out: enc: cipher text
		enc_len: cipher text length
*/
long hbkm_pkey_encrypt(key_serial_t key_id,
			const char *info,
			const void *data, size_t data_len,
			void *enc, size_t enc_len);

/*
	in: key_id: key serial id
		info: ref to hbkm_pkey_query() info param
		enc: cipher text
		enc_len: cipher text length
	out: data: plain text
		data_len: plain text length
*/
long hbkm_pkey_decrypt(key_serial_t key_id,
			const char *info,
			const void *enc, size_t enc_len,
			void *data, size_t data_len);

/*
	in: key_id: key serial id
		info: ref to hbkm_pkey_query() info param
		data: plain text
		data_len: plain text length
	out: sig: signature
		sig_len: signature length
*/
long hbkm_pkey_sign(key_serial_t key_id,
			const char *info,
			const void *data, size_t data_len,
			void *sig, size_t sig_len);

/*
	in: key_id: key serial id
		data: plain text
		data_len: plain text length
		sig: signature
		sig_len: signature length
	out: void
*/
long hbkm_pkey_verify(key_serial_t key_id,
			const char *info,
			const void *data, size_t data_len,
			const void *sig, size_t sig_len);
