/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */
#ifndef UCC_SCHEDULE_PIPELINED_H_
#define UCC_SCHEDULE_PIPELINED_H_
#include "components/base/ucc_base_iface.h"

#define UCC_SCHEDULE_FRAG_MAX_TASKS 8

typedef struct ucc_schedule_pipelined ucc_schedule_pipelined_t;

#define UCC_SCHEDULE_PIPELINED_MAX_FRAGS 4

/* frag_init is the callback provided by the user of pipelined
   framework (e.g., TL that needs to build a pipeline) that is reponsible
   for allocation of a single fragment schedule  */
typedef ucc_status_t (*ucc_schedule_frag_init_fn_t)(
    ucc_base_coll_args_t *coll_args, ucc_schedule_pipelined_t *schedule_p,
    ucc_base_team_t *team, ucc_schedule_t **frag);

/* frag setup is the callback which is triggered whenever a fragment is
   being re-launched (or launched for the first time ).
   This callback is used to update coll_args values (ptr offsets, counts)
   depending on the frag num.

   frag_num - is the sequencial number of currently launched fragment. */
typedef ucc_status_t (*ucc_schedule_frag_setup_fn_t)(
    ucc_schedule_pipelined_t *schedule_p, ucc_schedule_t *frag, int frag_num);

typedef struct ucc_schedule_pipelined {
    ucc_schedule_t               super;
    /* Array of the frag schedules - 1 schedule per pipeline entry */
    ucc_schedule_t *             frags[UCC_SCHEDULE_PIPELINED_MAX_FRAGS];
    /* n_frags - is the depth of the pipeline, ie how many fragments can
       be outstanding at a time */
    int                          n_frags;
    /* total number of fragments started so far. Note, total number of frags
       to be executed is stored in super.n_tasks */
    int                          n_frags_started;
    /* number of frags active in the pipeline */
    int                          n_frags_in_pipeline;
    /* sequential flag. if set to 1 the pipeline sets additional deps
       between the tasks in different frags. This prevents out-of-order
       task launch in different frags of a pipeline */
    int                          sequential;
    int                          next_frag_to_post;
    ucc_schedule_frag_setup_fn_t frag_setup;
    ucc_recursive_spinlock_t     lock;
} ucc_schedule_pipelined_t;

/* Creates a pipelined schedule for the algorithm defined by "frag_init".

   frag_init, frag_setup - client callbacks used to init the pipeline.
   n_frags - pipeline depth
   n_frags_total - total number of fragments to be launched. If n_frags_total
   > n_frags, then some frag schedules will be re-launched multiple times. */
ucc_status_t ucc_schedule_pipelined_init(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_schedule_frag_init_fn_t  frag_init,
    ucc_schedule_frag_setup_fn_t frag_setup, int n_frags, int n_frags_total,
    int sequential, ucc_schedule_pipelined_t *schedule_p);

ucc_status_t ucc_schedule_pipelined_post(ucc_coll_task_t *task);

ucc_status_t ucc_schedule_pipelined_finalize(ucc_coll_task_t *task);
#endif
