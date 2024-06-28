/*
 * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 *
 */

#ifndef UROM_UCC_H_
#define UROM_UCC_H_

#include <ucp/api/ucp.h>
#include <ucc/api/ucc.h>

#ifdef __cplusplus
extern "C" {
#endif

/* UCC serializing next raw, iter points to the offset place and returns the
   buffer start */
#define urom_ucc_serialize_next_raw(_iter, _type, _offset) \
    ({                                                     \
        _type *_result = (_type *)(*(_iter));              \
        *(_iter) = UCS_PTR_BYTE_OFFSET(*(_iter), _offset); \
        _result;                                           \
    })

/* UCC command types */
enum urom_worker_ucc_cmd_type {
    UROM_WORKER_CMD_UCC_LIB_CREATE,                  /* UCC library create command */
    UROM_WORKER_CMD_UCC_LIB_DESTROY,                 /* UCC library destroy command */
    UROM_WORKER_CMD_UCC_CONTEXT_CREATE,              /* UCC context create command */
    UROM_WORKER_CMD_UCC_CONTEXT_DESTROY,             /* UCC context destroy command */
    UROM_WORKER_CMD_UCC_TEAM_CREATE,                 /* UCC team create command */
    UROM_WORKER_CMD_UCC_COLL,                        /* UCC collective create command */
    UROM_WORKER_CMD_UCC_CREATE_PASSIVE_DATA_CHANNEL, /* UCC passive data channel command */
};

/*
 * UCC library create command structure
 *
 * Input parameters for creating the library handle. The semantics of the
 * parameters are defined by ucc.h On successful completion of
 * urom_worker_cmd_ucc_lib_create, The UROM worker will generate a notification
 * on the notification queue. This notification has reference to local library
 * handle on the worker. The implementation can choose to create shadow handles
 * or safely pack the library handle on the BlueCC worker to the AEU.
 */
struct urom_worker_cmd_ucc_lib_create {
    void *params; /* UCC library parameters */
};

/* UCC context create command structure */
struct urom_worker_cmd_ucc_context_create {
    union {
        int64_t  start; /* The started index */
        int64_t *array; /* Set stride to <= 0 if array is used */
    };
    int64_t  stride;  /* Set number of strides */
    int64_t  size;    /* Set stride size */
    void    *base_va; /* Shared buffer address */
    uint64_t len;     /* Buffer length */
};

/* UCC passive data channel command structure */
struct urom_worker_cmd_ucc_pass_dc {
    void  *ucp_addr; /* UCP worker address on host */
    size_t addr_len; /* UCP worker address length */
};

/* UCC context destroy command structure */
struct urom_worker_cmd_ucc_context_destroy {
    ucc_context_h context_h; /* UCC context pointer */
};

/* UCC team create command structure */
struct urom_worker_cmd_ucc_team_create {
    int64_t       start;     /* Team start index */
    int64_t       stride;    /* Number of strides */
    int64_t       size;      /* Stride size */
    ucc_context_h context_h; /* UCC context */
};

/* UCC collective command structure */
struct urom_worker_cmd_ucc_coll {
    ucc_coll_args_t *coll_args;        /* Collective arguments */
    ucc_team_h       team;             /* UCC team */
    int              use_xgvmi;        /* If operation uses XGVMI */
    void            *work_buffer;      /* Work buffer */
    size_t           work_buffer_size; /* Buffer size */
    size_t           team_size;        /* Team size */
};

/* UROM UCC worker command structure */
struct urom_worker_ucc_cmd {
    enum urom_worker_ucc_cmd_type cmd_type;
    uint64_t dpu_worker_id; /* DPU worker id as part of the team */
    union {
        struct urom_worker_cmd_ucc_lib_create      lib_create_cmd;      /* Lib create command */
        struct urom_worker_cmd_ucc_context_create  context_create_cmd;  /* Context create command */
        struct urom_worker_cmd_ucc_context_destroy context_destroy_cmd; /* Context destroy command */
        struct urom_worker_cmd_ucc_team_create     team_create_cmd;     /* Team create command */
        struct urom_worker_cmd_ucc_coll            coll_cmd;            /* UCC collective command */
        struct urom_worker_cmd_ucc_pass_dc         pass_dc_create_cmd;  /* Passive data channel command */
    };
};

/* UCC notification types */
enum urom_worker_ucc_notify_type {
    UROM_WORKER_NOTIFY_UCC_LIB_CREATE_COMPLETE,           /* Create UCC library on DPU notification */
    UROM_WORKER_NOTIFY_UCC_LIB_DESTROY_COMPLETE,          /* Destroy UCC library on DPU notification */
    UROM_WORKER_NOTIFY_UCC_CONTEXT_CREATE_COMPLETE,       /* Create UCC context on DPU notification */
    UROM_WORKER_NOTIFY_UCC_CONTEXT_DESTROY_COMPLETE,      /* Destroy UCC context on DPU notification */
    UROM_WORKER_NOTIFY_UCC_TEAM_CREATE_COMPLETE,          /* Create UCC team on DPU notification */
    UROM_WORKER_NOTIFY_UCC_COLLECTIVE_COMPLETE,           /* UCC collective completion notification */
    UROM_WORKER_NOTIFY_UCC_PASSIVE_DATA_CHANNEL_COMPLETE, /* UCC data channel completion notification */
};

/* UCC context create notification structure */
struct urom_worker_ucc_notify_context_create {
    ucc_context_h context; /* Pointer to UCC context */
};

/* UCC team create notification structure */
struct urom_worker_ucc_notify_team_create {
    ucc_team_h team; /* Pointer to UCC team */
};

/* UCC collective notification structure */
struct urom_worker_ucc_notify_collective {
    ucc_status_t status; /* UCC collective status */
};

/* UCC passive data channel notification structure */
struct urom_worker_ucc_notify_pass_dc {
    ucc_status_t status; /* UCC  data channel status */
};

/* UROM UCC worker notification structure */
struct urom_worker_notify_ucc {
    enum urom_worker_ucc_notify_type notify_type;
    uint64_t dpu_worker_id; /* DPU worker id */
    union {
        struct urom_worker_ucc_notify_context_create context_create_nqe; /* Context create notification */
        struct urom_worker_ucc_notify_team_create    team_create_nqe;    /* Team create notification */
        struct urom_worker_ucc_notify_collective     coll_nqe;           /* Collective notification */
        struct urom_worker_ucc_notify_pass_dc        pass_dc_nqe;        /* Passive data channel notification */
    };
};

typedef struct ucc_worker_key_buf {
    size_t src_len;
    size_t dst_len;
    char   rkeys[1024];
} ucc_worker_key_buf;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* UROM_UCC_H_ */
