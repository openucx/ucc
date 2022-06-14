#include "ucc_mpool.h"
#include "ucc_malloc.h"
#include "ucc_log.h"

static ucc_mpool_ops_t ucc_default_mpool_ops = {
    .chunk_alloc   = ucc_mpool_hugetlb_malloc,
    .chunk_release = ucc_mpool_hugetlb_free,
    .obj_init      = NULL,
    .obj_cleanup   = NULL
};

static ucs_status_t
ucc_mpool_chunk_alloc_wrapper(ucs_mpool_t *mp, size_t *size_p, void **chunk_p)
{
    ucc_mpool_t *mpool = ucc_derived_of(mp, ucc_mpool_t);
    ucc_status_t st;

    st = mpool->ucc_ops->chunk_alloc(mpool, size_p, chunk_p);
    return ucc_status_to_ucs_status(st);
}

static void ucc_mpool_chunk_release_wrapper(ucs_mpool_t *mp, void *chunk)
{
    ucc_mpool_t *mpool = ucc_derived_of(mp, ucc_mpool_t);
    mpool->ucc_ops->chunk_release(mpool, chunk);
}

static void ucc_mpool_obj_init_wrapper(ucs_mpool_t *mp, void *obj, void *chunk)
{
    ucc_mpool_t *mpool = ucc_derived_of(mp, ucc_mpool_t);
    mpool->ucc_ops->obj_init(mpool, obj, chunk);
}

static void ucc_mpool_obj_cleanup_wrapper(ucs_mpool_t *mp, void *obj)
{
    ucc_mpool_t *mpool = ucc_derived_of(mp, ucc_mpool_t);
    mpool->ucc_ops->obj_cleanup(mpool, obj);
}

ucc_status_t ucc_mpool_init(ucc_mpool_t *mp, size_t priv_size, size_t elem_size,
                            size_t align_offset, size_t alignment,
                            unsigned elems_per_chunk, unsigned max_elems,
                            ucc_mpool_ops_t *ops, ucc_thread_mode_t tm,
                            const char *name)
{
    ucs_mpool_ops_t *ucs_ops = ucc_calloc(1, sizeof(*ucs_ops), "mpool_ops");
#if UCS_HAVE_MPOOL_PARAMS
    ucs_mpool_params_t params;
#endif

    if (!ucs_ops) {
        ucc_error("failed to allocate %zd bytes for mpool ucs ops",
                  sizeof(*ucs_ops));
        return UCC_ERR_NO_MEMORY;
    }

    ucc_spinlock_init(&mp->lock, 0);
    mp->tm                 = tm;
    mp->ucc_ops            = ops ? ops : &ucc_default_mpool_ops;
    ucs_ops->chunk_alloc   = ucc_mpool_chunk_alloc_wrapper;
    ucs_ops->chunk_release = ucc_mpool_chunk_release_wrapper;
    if (mp->ucc_ops->obj_init != NULL) {
        ucs_ops->obj_init = ucc_mpool_obj_init_wrapper;
    }
    if (mp->ucc_ops->obj_cleanup != NULL) {
        ucs_ops->obj_cleanup = ucc_mpool_obj_cleanup_wrapper;
    }
#if UCS_HAVE_MPOOL_PARAMS
    ucs_mpool_params_reset(&params);
    params.priv_size       = priv_size;
    params.elem_size       = elem_size;
    params.align_offset    = align_offset;
    params.alignment       = alignment;
    params.malloc_safe     = 0;
    params.elems_per_chunk = elems_per_chunk;
    params.max_chunk_size  = SIZE_MAX;
    params.max_elems       = max_elems;
    params.grow_factor     = 1.0;
    params.ops             = ucs_ops;
    params.name            = name;

    return ucs_status_to_ucc_status(ucs_mpool_init(&params, &mp->super));
#else
    return ucs_status_to_ucc_status(
        ucs_mpool_init(&mp->super, priv_size, elem_size, align_offset,
                       alignment, elems_per_chunk, max_elems, ucs_ops, name));
#endif
}

void ucc_mpool_cleanup(ucc_mpool_t *mp, int leak_check)
{
    void *ops = (void*)mp->super.data->ops;

    ucs_mpool_cleanup(&mp->super, leak_check);
    ucc_free(ops);
    ucc_spinlock_destroy(&mp->lock);
}

ucc_status_t ucc_mpool_hugetlb_malloc(ucc_mpool_t *mp, size_t *size_p,
                                      void **chunk_p)
{
    ucs_status_t st;

    st = ucs_mpool_hugetlb_malloc(&mp->super, size_p, chunk_p);
    return ucs_status_to_ucc_status(st);
}

void ucc_mpool_hugetlb_free(ucc_mpool_t *mp, void *chunk)
{
    ucs_mpool_hugetlb_free(&mp->super, chunk);
}
