/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/


/**
 * Construct a UCC version identifier from major and minor version numbers.
 */
#define UCC_VERSION(_major, _minor) \
	(((_major) << UCC_VERSION_MAJOR_SHIFT) | \
	 ((_minor) << UCC_VERSION_MINOR_SHIFT))
#define UCC_VERSION_MAJOR_SHIFT    24
#define UCC_VERSION_MINOR_SHIFT    16


/**
 * UCC API version is 1.0
 */
#define UCC_API_MAJOR    1
#define UCC_API_MINOR    0
#define UCC_API_VERSION  UCC_VERSION(1, 0)
