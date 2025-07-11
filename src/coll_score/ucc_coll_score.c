/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_coll_score.h"
#include "utils/ucc_string.h"
#include "utils/ucc_log.h"
#include "utils/ucc_coll_utils.h"

char *ucc_score_to_str(ucc_score_t score, char *buf, size_t max) {
    if (score == UCC_SCORE_MAX) {
        ucc_strncpy_safe(buf, "inf", max);
    } else {
        ucc_snprintf_safe(buf, max, "%d", score);
    }

    return buf;
}

ucc_status_t ucc_coll_score_alloc(ucc_coll_score_t **score)
{
    ucc_coll_score_t *s = ucc_malloc(sizeof(*s), "ucc_coll_score");
    int               i, j;
    if (!s) {
        ucc_error("failed to allocate %zd bytes for ucc_coll_score",
                  sizeof(*s));
        *score = NULL;
        return UCC_ERR_NO_MEMORY;
    }
    for (i = 0; i < UCC_COLL_TYPE_NUM; i++) {
        for (j = 0; j < UCC_MEMORY_TYPE_LAST; j++) {
            ucc_list_head_init(&s->scores[i][j]);
        }
    }
    *score = s;
    return UCC_OK;
}

static inline void ucc_msg_range_free(ucc_msg_range_t *r)
{
    ucc_list_destruct(&r->fallback, ucc_coll_entry_t, ucc_free, list_elem);
    ucc_free(r);
}

static inline ucc_status_t
coll_score_add_range(ucc_coll_score_t *score, ucc_coll_type_t coll_type,
                     ucc_memory_type_t mem_type, size_t start, size_t end,
                     ucc_score_t msg_score, ucc_base_coll_init_fn_t init,
                     ucc_base_team_t *team)
{
    ucc_msg_range_t *r;
    ucc_msg_range_t *range;
    ucc_list_link_t *list, *insert_pos, *next;

    if (start >= end) {
        return UCC_ERR_INVALID_PARAM;
    }
    r = ucc_malloc(sizeof(*r), "ucc_msg_range");
    if (!r) {
        ucc_error("failed to allocate %zd bytes for ucc_msg_range", sizeof(*r));
        return UCC_ERR_NO_MEMORY;
    }

    ucc_list_head_init(&r->fallback);

    r->start       = start;
    r->end         = end;
    r->super.score = msg_score;
    r->super.init  = init;
    r->super.team  = team;
    list           = &score->scores[ucc_ilog2(coll_type)][mem_type];
    insert_pos     = list;
    ucc_list_for_each(range, list, super.list_elem) {
        if (start >= range->end) {
            insert_pos = &range->super.list_elem;
        } else {
            break;
        }
    }
    ucc_list_insert_after(insert_pos, &r->super.list_elem);
    /*sanity check: ranges shuold not overlap */
    next = r->super.list_elem.next;
    if ((next != list) &&
        (ucc_container_of(next, ucc_msg_range_t, super.list_elem)->start <
         r->end)) {
        ucc_error("attempt to add overlaping range");
        ucc_list_del(&r->super.list_elem);
        ucc_msg_range_free(r);
        return UCC_ERR_INVALID_PARAM;
    }
    return UCC_OK;
}
ucc_status_t ucc_coll_score_add_range(ucc_coll_score_t *score,
                                      ucc_coll_type_t   coll_type,
                                      ucc_memory_type_t mem_type, size_t start,
                                      size_t end, ucc_score_t msg_score,
                                      ucc_base_coll_init_fn_t init,
                                      ucc_base_team_t *       team)
{
    if (msg_score == 0) {
        /* score 0 means range is disabled, just skip */
        return UCC_OK;
    }
    return coll_score_add_range(score, coll_type, mem_type, start, end,
                                msg_score, init, team);
}

void ucc_coll_score_free(ucc_coll_score_t *score)
{
    int i, j;
    if (!score) {
        return;
    }
    for (i = 0; i < UCC_COLL_TYPE_NUM; i++) {
        for (j = 0; j < UCC_MEMORY_TYPE_LAST; j++) {
            ucc_list_destruct(&score->scores[i][j], ucc_msg_range_t,
                              ucc_msg_range_free, super.list_elem);
        }
    }
    ucc_free(score);
}

static ucc_status_t ucc_fallback_alloc(ucc_score_t              score,
                                       ucc_base_coll_init_fn_t  init,
                                       ucc_base_team_t         *team,
                                       ucc_coll_entry_t       **_fb)
{
    ucc_coll_entry_t *fb;

    fb = ucc_malloc(sizeof(*fb), "fallback");
    if (ucc_unlikely(!fb)) {
        ucc_error("failed to allocate %zd bytes for fallback",
                  sizeof(ucc_coll_entry_t));
        *_fb = NULL;
        return UCC_ERR_NO_MEMORY;
    }
    fb->score = score;
    fb->init  = init;
    fb->team  = team;
    *_fb      = fb;
    return UCC_OK;
}

static inline void ucc_fallback_insert(ucc_list_link_t  *list,
                                       ucc_coll_entry_t *fb)
{
    ucc_list_link_t  *insert_pos;
    ucc_coll_entry_t *f;

    insert_pos = list;
    ucc_list_for_each(f, list, list_elem) {
        if (fb->score == f->score && fb->init == f->init &&
            fb->team == f->team) {
            ucc_free(fb);
            /* same fallback: skip */
            return;
        }
        if (fb->score < f->score) {
            insert_pos = &f->list_elem;
        } else {
            break;
        }
    }
    ucc_list_insert_after(insert_pos, &fb->list_elem);
}

#define FB_ALLOC_INSERT(_fb_in, _fb_out, _dest, _status, _label) do {   \
        _status =                                                       \
            ucc_fallback_alloc((_fb_in)->score, (_fb_in)->init,         \
                               (_fb_in)->team, &(_fb_out));             \
        if (ucc_unlikely(UCC_OK != _status)) {                          \
            goto _label;                                                \
        }                                                               \
        ucc_fallback_insert(&(_dest)->fallback, _fb_out);               \
    } while (0)

static ucc_status_t ucc_fallback_copy(const ucc_msg_range_t *in,
                                      ucc_msg_range_t       *out)
{
    ucc_status_t      status = UCC_OK;
    ucc_coll_entry_t *fb_in, *fb;

    ucc_list_for_each(fb_in, &in->fallback, list_elem) {
        FB_ALLOC_INSERT(fb_in, fb, out, status, out);
    }
out:
    return status;
}

static ucc_status_t ucc_msg_range_dup(const ucc_msg_range_t *in,
                                      ucc_msg_range_t      **out)
{
    ucc_msg_range_t *r;
    ucc_status_t status;

    r = ucc_malloc(sizeof(*r), "msg_range");
    if (ucc_unlikely(!r)) {
        *out = NULL;
        ucc_error("failed to allocate %zd bytes for msgrange", sizeof(*r));
        return UCC_ERR_NO_MEMORY;
    }
    memcpy(r, in, sizeof(*r));
    ucc_list_head_init(&r->fallback);
    status = ucc_fallback_copy(in, r);
    if (status != UCC_OK) {
        ucc_msg_range_free(r);
        r = NULL;
    }
    *out = r;
    return status;
}

#define MSG_RANGE_DUP(_r)                                                      \
    ({                                                                         \
        ucc_msg_range_t *_dup;                                                 \
        status = ucc_msg_range_dup(_r, &_dup);                                 \
        if (UCC_OK != status) {                                                \
            goto out;                                                          \
        }                                                                      \
        _dup;                                                                  \
    })

static ucc_status_t ucc_msg_range_add_fallback(const ucc_msg_range_t *in,
                                               ucc_msg_range_t       *out)
{
    ucc_coll_entry_t *fb;
    ucc_status_t      status;

    if (in->super.init == out->super.init &&
        in->super.team == out->super.team) {
        return UCC_OK;
    }

    status = ucc_fallback_alloc(in->super.score, in->super.init, in->super.team,
                                &fb);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }
    status = ucc_fallback_copy(in, out);
    ucc_fallback_insert(&out->fallback, fb);
    return status;
}

#define ADD_FALLBACK(_in, _out)                                                \
    do {                                                                       \
        status = ucc_msg_range_add_fallback(_in, _out);                        \
        if (UCC_OK != status) {                                                \
            goto out;                                                          \
        }                                                                      \
    } while (0)

static ucc_status_t ucc_score_list_dup(const ucc_list_link_t *src,
                                       ucc_list_link_t       *dst)
{
    ucc_msg_range_t *range, *r;
    ucc_status_t     status;

    ucc_list_head_init(dst);
    ucc_list_for_each(range, src, super.list_elem) {
        r = MSG_RANGE_DUP(range);
        ucc_list_add_tail(dst, &r->super.list_elem);
    }
    return UCC_OK;
out:
    ucc_list_for_each_safe(range, r, dst, super.list_elem) {
        ucc_list_del(&range->super.list_elem);
        ucc_msg_range_free(range);
    }
    ucc_assert(ucc_list_is_empty(dst));
    return UCC_ERR_NO_MEMORY;
}

static inline int ucc_msg_range_fb_compare(ucc_msg_range_t *r1,
                                           ucc_msg_range_t *r2)
{
    ucc_list_link_t  *l1, *l2;
    ucc_coll_entry_t *fb1, *fb2;

    l1 = &r1->fallback;
    l2 = &r2->fallback;

    if (ucc_list_length(l1) != ucc_list_length(l2)) {
        return 0;
    }

    fb2 = ucc_list_head(l2, ucc_coll_entry_t, list_elem);
    ucc_list_for_each(fb1, l1, list_elem) {
        if (fb1->score != fb2->score || fb1->init != fb2->init ||
            fb1->team != fb2->team) {
            return 0;
        }
        fb2 = ucc_list_next(&fb2->list_elem, ucc_coll_entry_t, list_elem);
    }
    return 1;
}

static ucc_status_t ucc_coll_score_merge_one(ucc_list_link_t *list1,
                                             ucc_list_link_t *list2,
                                             ucc_list_link_t *out)
{
    ucc_list_link_t  lst1, lst2;
    ucc_msg_range_t *r1, *r2, *left, *right, *best, *new;
    ucc_msg_range_t *range, *temp, *next;
    ucc_status_t     status;

    if (ucc_list_is_empty(list1) && ucc_list_is_empty(list2)) {
        return UCC_OK;
    } else if (ucc_list_is_empty(list1)) {
        return ucc_score_list_dup(list2, out);
    } else if (ucc_list_is_empty(list2)) {
        return ucc_score_list_dup(list1, out);
    }
    /* list1 and list2 both non-empty: need to intersect ranges */
    status = ucc_score_list_dup(list1, &lst1);
    if (UCC_OK != status) {
        return status;
    }
    status = ucc_score_list_dup(list2, &lst2);
    if (UCC_OK != status) {
        goto out;
    }

    while (!(ucc_list_is_empty(&lst1) && ucc_list_is_empty(&lst2))) {
        if (ucc_list_is_empty(&lst1)) {
            ucc_list_add_tail(out, &(ucc_list_extract_head(&lst2,
                                     ucc_coll_entry_t, list_elem)->list_elem));
            continue;
        }
        if (ucc_list_is_empty(&lst2)) {
            ucc_list_add_tail(out, &(ucc_list_extract_head(&lst1,
                                     ucc_coll_entry_t, list_elem)->list_elem));
            continue;
        }
        r1   = ucc_list_head(&lst1, ucc_msg_range_t, super.list_elem);
        r2   = ucc_list_head(&lst2, ucc_msg_range_t, super.list_elem);
        left = (r1->start < r2->start) ? r1 : r2; //NOLINT

        if (r1->start == r2->start) {
            if (r1->end == r2->end) {
                best = (r1->super.score > r2->super.score) ? r1 : r2;
                left                  = (best == r1) ? r2 : r1;
                ucc_list_del(&r1->super.list_elem);
                ucc_list_del(&r2->super.list_elem);
                ucc_list_add_tail(out, &best->super.list_elem);
                ADD_FALLBACK(left, best);
                ucc_msg_range_free(left);
                continue;
            }
            left = (r1->end < r2->end) ? r1 : r2;
            right        = (left == r1) ? r2 : r1;
            new          = MSG_RANGE_DUP(right);
            right->start = new->end = left->end;
            ucc_list_del(&left->super.list_elem);
            if (left->super.score < new->super.score) {
                SWAP(left, new, void *);
            }
            ADD_FALLBACK(new, left);
            ucc_msg_range_free(new);

            ucc_list_add_tail(out, &left->super.list_elem);
            continue;
        }
        right = (left == r1) ? r2 : r1;
        if (left->end <= right->start) {
            /* ranges don't overlap - copy over */
            ucc_list_del(&left->super.list_elem);
            ucc_list_add_tail(out, &left->super.list_elem);
        } else {
            new      = MSG_RANGE_DUP(left);
            new->end = right->start;
            ucc_list_add_tail(out, &new->super.list_elem);
            left->start = right->start;
        }
    }
    /* Merge consequtive ranges with the same score, same init fn,
       same team and same fallback sequence
       if any have been produced by the algorithm above */

    ucc_list_for_each_safe(range, temp, out, super.list_elem) {
        if (range->super.list_elem.next != out) {
            next = ucc_container_of(range->super.list_elem.next,
                                    ucc_msg_range_t, super.list_elem);
            if (range->super.score == next->super.score &&
                range->end == next->start &&
                range->super.init == next->super.init &&
                range->super.team == next->super.team &&
                1 == ucc_msg_range_fb_compare(range, next)) {
                next->start = range->start;
                ucc_list_del(&range->super.list_elem);
                ucc_msg_range_free(range);
            }
        }
    }
    return UCC_OK;

out:
    ucc_list_destruct(&lst2, ucc_msg_range_t, ucc_msg_range_free,
                      super.list_elem);
    ucc_list_destruct(&lst1, ucc_msg_range_t, ucc_msg_range_free,
                      super.list_elem);
    ucc_list_destruct(out, ucc_msg_range_t, ucc_msg_range_free,
                      super.list_elem);
    return status;
}

ucc_status_t ucc_coll_score_merge(ucc_coll_score_t * score1,
                                  ucc_coll_score_t * score2,
                                  ucc_coll_score_t **rst, int free_inputs)
{
    ucc_coll_score_t *out;
    ucc_status_t      status;
    int               i, j;
    status = ucc_coll_score_alloc(&out);
    if (UCC_OK != status) {
        goto out;
    }
    for (i = 0; i < UCC_COLL_TYPE_NUM; i++) {
        for (j = 0; j < UCC_MEMORY_TYPE_LAST; j++) {
            status = ucc_coll_score_merge_one(&score1->scores[i][j],
                                              &score2->scores[i][j],
                                              &out->scores[i][j]);
            if (UCC_OK != status) {
                ucc_coll_score_free(out);
                goto out;
            }
        }
    }
    *rst = out;
out:
    if (free_inputs) {
        ucc_coll_score_free(score1);
        ucc_coll_score_free(score2);
    }
    return status;
}

 ucc_status_t ucc_coll_score_merge_in(ucc_coll_score_t **dst,
                                     ucc_coll_score_t *src)
{
    ucc_coll_score_t *tmp = NULL;
    ucc_status_t      status;

    status = ucc_coll_score_merge(src, *dst, &tmp, 1);
    *dst = tmp;
    return status;
}

static ucc_status_t str_to_coll_type(const char *str, unsigned *ct_n,
                                     ucc_coll_type_t **ct)
{
    ucc_status_t    status = UCC_OK;
    char          **tokens;
    unsigned        i, n_tokens;
    ucc_coll_type_t t;
    tokens = ucc_str_split(str, ",");
    if (!tokens) {
        status = UCC_ERR_INVALID_PARAM;
        goto out;
    }
    n_tokens = ucc_str_split_count(tokens);
    *ct      = ucc_malloc(n_tokens * sizeof(ucc_coll_type_t), "ucc_coll_types");
    if (!(*ct)) {
        ucc_error("failed to allocate %zd bytes for ucc_coll_types",
                  sizeof(ucc_coll_type_t) * n_tokens);
        status = UCC_ERR_NO_MEMORY;
        goto out;
    }
    *ct_n = 0;
    for (i = 0; i < n_tokens; i++) {
        t = ucc_coll_type_from_str(tokens[i]);
        if (t == UCC_COLL_TYPE_LAST) {
            /* entry does not match any coll type name */
            ucc_free(*ct);
            *ct    = NULL;
            status = UCC_ERR_NOT_FOUND;
            goto out;
        }
        (*ct)[*ct_n] = t;
        (*ct_n)++;
    }
out:
    ucc_str_split_free(tokens);
    return status;
}

static ucc_status_t str_to_score(const char *str, ucc_score_t *score)
{
    if (0 == strcasecmp("inf", str)) {
        *score = UCC_SCORE_MAX;
    } else if (UCC_OK != ucc_str_is_number(str)) {
        return UCC_ERR_NOT_FOUND;
    } else {
        *score = (ucc_score_t)atoi(str);
    }
    return UCC_OK;
}

static ucc_status_t str_to_alg_id(const char *str, const char **alg_id)
{
    if ('@' != str[0]) {
        return UCC_ERR_NOT_FOUND;
    }
    *alg_id = str + 1;
    return UCC_OK;
}

static ucc_status_t str_to_msgranges(const char *str, size_t **ranges,
                                     unsigned *n_ranges)
{
    ucc_status_t      status = UCC_OK;
    char            **tokens;
    char            **tokens2;
    unsigned          i, n_tokens, n_tokens2;
    size_t            m1, m2;
    tokens = ucc_str_split(str, ",");
    if (!tokens) {
        status = UCC_ERR_INVALID_PARAM;
        goto out;
    }
    n_tokens = ucc_str_split_count(tokens);
    *ranges = ucc_malloc(2 * n_tokens * sizeof(size_t), "ucc_msgsize_ranges");
    if (!(*ranges)) {
        ucc_error("failed to allocate %zd bytes for ucc_msgsize_ranges",
                  sizeof(size_t) * 2 * n_tokens);
        status = UCC_ERR_NO_MEMORY;
        goto out;
    }

    for (i = 0; i < n_tokens; i++) {
        tokens2 = ucc_str_split(tokens[i], "-");
        if (!tokens2) {
            goto err;
        }
        n_tokens2 = ucc_str_split_count(tokens2);
        if (n_tokens2 != 2) {
            goto err;
        }
        if (UCC_OK != ucc_str_to_memunits(tokens2[0], &m1) ||
            UCC_OK != ucc_str_to_memunits(tokens2[1], &m2)) {
            goto err;
        }
        ucc_str_split_free(tokens2);
        (*ranges)[2 * i] = m1;
        (*ranges)[2 * i + 1] = m2;
    }
    *n_ranges = i;
out:
    ucc_str_split_free(tokens);
    return status;
err:
    ucc_str_split_free(tokens2);
    ucc_free(*ranges);
    *ranges = NULL;
    status  = UCC_ERR_NOT_FOUND;
    goto out;
}

static ucc_status_t str_to_tsizes(const char *str, ucc_rank_t **tsizes,
                                  unsigned *n_tsizes)
{
    ucc_status_t status = UCC_OK;
    char **      tokens;
    char **      tokens2;
    unsigned     i, n_tokens, n_tokens2;

    /* team_size qualifer should be enclosed in "[]".
       It it a coma-separated list of ranges start-end or
       single values (exact team size) */
    if ('[' != str[0] || ']' != str[strlen(str) - 1]) {
        return UCC_ERR_NOT_FOUND;
    }
    tokens = ucc_str_split(str + 1, ",");
    if (!tokens) {
        status = UCC_ERR_INVALID_PARAM;
        goto out;
    }
    n_tokens = ucc_str_split_count(tokens);
    *tsizes = ucc_malloc(2 * n_tokens * sizeof(ucc_rank_t), "ucc_tsize_ranges");
    if (!(*tsizes)) {
        ucc_error("failed to allocate %zd bytes for ucc_tsize_ranges",
                  sizeof(ucc_rank_t) * 2 * n_tokens);
        status = UCC_ERR_NO_MEMORY;
        goto out;
    }
    ucc_assert(']' == tokens[n_tokens - 1][strlen(tokens[n_tokens - 1]) - 1]);
    /* remove last "]" from the parsed string, we have already checked it was present */
    tokens[n_tokens - 1][strlen(tokens[n_tokens - 1]) - 1] = '\0';
    for (i = 0; i < n_tokens; i++) {
        tokens2 = ucc_str_split(tokens[i], "-");
        if (!tokens2) {
            status  = UCC_ERR_INVALID_PARAM;
            goto err;
        }
        n_tokens2 = ucc_str_split_count(tokens2);
        if (n_tokens2 == 1) {
            /* exact team size - single value */
            if (UCC_OK != ucc_str_is_number(tokens2[0])) {
                status = UCC_ERR_INVALID_PARAM;
                goto err;
            }
            (*tsizes)[2 * i]     = (ucc_rank_t)atoi(tokens2[0]);
            (*tsizes)[2 * i + 1] = (ucc_rank_t)atoi(tokens2[0]);
        } else {
            if (n_tokens2 != 2) {
                status  = UCC_ERR_INVALID_PARAM;
                goto err;
            }
            if (UCC_OK == ucc_str_is_number(tokens2[0])) {
                (*tsizes)[2 * i] = (ucc_rank_t)atoi(tokens2[0]);
            } else {
                status  = UCC_ERR_INVALID_PARAM;
                goto err;
            }
            if (0 == strcasecmp("inf", tokens2[1])) {
                (*tsizes)[2 * i + 1] = UCC_RANK_MAX;
            } else if (UCC_OK == ucc_str_is_number(tokens2[1])) {
                (*tsizes)[2 * i + 1] = (ucc_rank_t)atoi(tokens2[1]);
            } else {
                status  = UCC_ERR_INVALID_PARAM;
                goto err;
            }
            if ((*tsizes)[2 * i + 1] < (*tsizes)[2 * i]) {
                status  = UCC_ERR_INVALID_PARAM;
                goto err;
            }
        }
        ucc_str_split_free(tokens2);
    }
    *n_tsizes = i;
out:
    ucc_str_split_free(tokens);
    return status;
err:
    ucc_str_split_free(tokens2);
    ucc_free(*tsizes);
    *tsizes = NULL;
    goto out;
}

static ucc_status_t ucc_coll_score_parse_str(const char *str,
                                             ucc_coll_score_t *score,
                                             ucc_rank_t team_size,
                                             ucc_base_coll_init_fn_t init,
                                             ucc_base_team_t *team,
                                             ucc_alg_id_to_init_fn_t alg_fn)
{
    ucc_status_t            status   = UCC_OK;
    ucc_coll_type_t        *ct       = NULL;
    size_t                 *msg      = NULL;
    ucc_rank_t             *tsizes   = NULL;
    ucc_base_coll_init_fn_t alg_init = NULL;
    const char*             alg_id   = NULL;
    ucc_score_t             score_v  = UCC_SCORE_INVALID;
    int                     ts_skip  = 0;
    uint32_t                mtypes   = 0;
    char                  **tokens;
    unsigned i, n_tokens, ct_n, c, m, n_ranges, r, n_tsizes;

    ct_n = n_ranges = n_tsizes = 0;
    tokens = ucc_str_split(str, ":");
    if (!tokens) {
        status = UCC_ERR_INVALID_PARAM;
        goto out;
    }
    n_tokens = ucc_str_split_count(tokens);
    for (i = 0; i < n_tokens; i++) {
        if (!ct && UCC_OK == str_to_coll_type(tokens[i], &ct_n, &ct)) {
            continue;
        }
        if (!mtypes && UCC_OK == ucc_str_to_mtype_map(tokens[i], ",",
                                                      &mtypes)) {
            continue;
        }
        if ((UCC_SCORE_INVALID == score_v) &&
            UCC_OK == str_to_score(tokens[i], &score_v)) {
            continue;
        }
        if (!msg && UCC_OK == str_to_msgranges(tokens[i], &msg, &n_ranges)) {
            continue;
        }
        if (!tsizes && UCC_OK == str_to_tsizes(tokens[i], &tsizes, &n_tsizes)) {
            continue;
        }
        if (!alg_id && UCC_OK == str_to_alg_id(tokens[i], &alg_id)) {
            continue;
        }
        /* if we get there then we could not match token to any field */
        status = UCC_ERR_INVALID_PARAM;
        ucc_error("failed to parse token \'%s\' in \'%s\'", tokens[i], str);
        //TODO add parsing of msg ranges and team size ranges
        goto out;
    }
    if (tsizes) {
        /* Team size qualifier was provided: check if we should apply this
           str setting to the current team */
        ts_skip = 1;
        for (i = 0; i < n_tsizes; i++) {
            if (team_size >= tsizes[2 * i] && team_size <= tsizes[2 * i + 1]) {
                ts_skip = 0;
                break;
            }
        }
    }
    if (!ts_skip && (UCC_SCORE_INVALID != score_v || NULL != alg_id)) {
        /* Score provided but not coll_types/mem_types.
           This means: apply score to ALL coll_types/mem_types */
        if (!ct) {
            ct_n = UCC_COLL_TYPE_NUM;
        }
        if (!mtypes) {
            mtypes = UCC_MEM_TYPE_MASK_FULL;
        }
        if (!msg) {
            n_ranges = 1;
        }
        for (c = 0; c < ct_n; c++) {
            for (m = 0; m < UCC_MEMORY_TYPE_LAST; m++) {
                if (!(UCC_BIT(m) & mtypes)) {
                    continue;
                }
                ucc_coll_type_t   coll_type = ct ? ct[c] :
                                                   (ucc_coll_type_t)UCC_BIT(c);
                ucc_memory_type_t mem_type  = (ucc_memory_type_t)m;
                if (alg_id) {
                    if (!alg_fn) {
                        status = UCC_ERR_NOT_SUPPORTED;
                        ucc_error("modifying algorithm id is not supported by "
                                  "component %s",
                                  team->context->lib->log_component.name);
                        goto out;
                    }
                    ucc_assert(NULL != team);
                    const char *alg_id_str = NULL;
                    int         alg_id_n   = 0;
                    if (UCC_OK == ucc_str_is_number(alg_id)) {
                        alg_id_n = atoi(alg_id);
                    } else {
                        alg_id_str = alg_id;
                    }
                    status = alg_fn(alg_id_n, alg_id_str, coll_type, mem_type,
                                    &alg_init);
                    if (UCC_ERR_INVALID_PARAM == status) {
                        ucc_error("incorrect algorithm id provided: %s, %s, "
                                  "component %s",
                                  alg_id, str,
                                  team->context->lib->log_component.name);
                        goto out;
                    } else if (UCC_ERR_NOT_SUPPORTED == status) {
                        ucc_error("modifying algorithm id is not supported for "
                                  "%s, alg %s, component %s",
                                  ucc_coll_type_str(coll_type), alg_id,
                                  team->context->lib->log_component.name);
                        goto out;
                    } else if (status < 0) {
                        ucc_error("failed to map alg id to init: %s, %s, "
                                  "status %s, component %s",
                                  alg_id, str, ucc_status_string(status),
                                  team->context->lib->log_component.name);
                        goto out;
                    }
                }
                for (r = 0; r < n_ranges; r++) {
                    size_t m_start = 0;
                    size_t m_end   = UCC_MSG_MAX;
                    if (msg) {
                        m_start = msg[r * 2];
                        m_end   = msg[r * 2 + 1];
                    }
                    status = coll_score_add_range(
                        score, coll_type, mem_type, m_start, m_end, score_v,
                        alg_init ? alg_init : init, team);
                }
            }
        }
    }
out:
    ucc_free(ct);
    ucc_free(msg);
    ucc_free(tsizes);
    ucc_str_split_free(tokens);
    return status;
}

ucc_status_t ucc_coll_score_alloc_from_str(const char *            str,
                                           ucc_coll_score_t **     score_p,
                                           ucc_rank_t              team_size,
                                           ucc_base_coll_init_fn_t init,
                                           ucc_base_team_t *       team,
                                           ucc_alg_id_to_init_fn_t alg_fn)
{
    ucc_coll_score_t *score;
    ucc_status_t      status;
    char            **tokens;
    unsigned          n_tokens, i;
    status = ucc_coll_score_alloc(&score);
    if (UCC_OK != status) {
        return status;
    }
    tokens = ucc_str_split(str, "#");
    if (!tokens) {
        status = UCC_ERR_INVALID_PARAM;
        goto error;
    }
    n_tokens = ucc_str_split_count(tokens);
    for (i = 0; i < n_tokens; i++) {
        status = ucc_coll_score_parse_str(tokens[i], score, team_size, init,
                                          team, alg_fn);
        if (UCC_OK != status) {
            goto error_msg;
        }
    }
    ucc_str_split_free(tokens);
    *score_p = score;
    return UCC_OK;
error_msg:
    ucc_error("failed to parse UCC_*_TUNE parameter: %s", tokens[i]);
error:
    *score_p = NULL;
    ucc_coll_score_free(score);
    ucc_str_split_free(tokens);
    return status;
}

static ucc_status_t ucc_coll_score_update_one(ucc_list_link_t *dest,
                                              ucc_list_link_t *src,
                                              ucc_score_t      default_score)
{
    ucc_list_link_t  *s = src->next;
    ucc_list_link_t  *d = dest->next;
    ucc_msg_range_t  *range, *tmp, *next, *rs, *rd, *new;
    ucc_coll_entry_t *fb;
    ucc_status_t      status;

    if (ucc_list_is_empty(src) && ucc_list_is_empty(dest)) {
        return UCC_OK;
    }

    while (s != src && d != dest) {
        rs = ucc_container_of(s, ucc_msg_range_t, super.list_elem);
        rd = ucc_container_of(d, ucc_msg_range_t, super.list_elem);
        ucc_assert((NULL == rs->super.init) || (NULL != rs->super.team));
        if (rd->start >= rs->end) {
            /* skip src range - no overlap */
            s = s->next;
            if (rs->super.init) {
                new = MSG_RANGE_DUP(rs);
                if (new->super.score == UCC_SCORE_INVALID) {
                    new->super.score = default_score;
                }
                ucc_list_insert_before(d, &new->super.list_elem);
            }
        } else if (rd->end <= rs->start) {
            /* no overlap - inverse case: skip dst range */
            d = d->next;
        } else if (rd->start <  rs->start) {
            new       = MSG_RANGE_DUP(rd);
            new->end  = rs->start;
            rd->start = rs->start;
            ucc_list_insert_before(d, &new->super.list_elem);
        } else if (rd->start > rs->start) {
            if (rs->super.init) {
                new = MSG_RANGE_DUP(rs);
                if (new->super.score == UCC_SCORE_INVALID) {
                    new->super.score = default_score;
                }
                new->end = rd->start;
                ucc_list_insert_before(d, &new->super.list_elem);
            }
            rs->start = rd->start;
        } else {
            /* same start */
            if (rs->end > rd->end) {
                if (UCC_SCORE_INVALID != rs->super.score) {
                    rd->super.score = rs->super.score;
                }
                if (rs->super.init) {
                    if (rs->super.init != rd->super.init) {
                        /* User setting overrides existing init fn. Save it as a fallback */
                        FB_ALLOC_INSERT(&rd->super, fb, rd, status, out);
                    }
                    rd->super.init = rs->super.init;
                    rd->super.team = rs->super.team;
                }
                rs->start = rd->end;
                d         = d->next;
            } else if (rs->end < rd->end) {
                new      = MSG_RANGE_DUP(rd);
                new->end = rs->end;
                if (UCC_SCORE_INVALID != rs->super.score) {
                    new->super.score = rs->super.score;
                }
                if (rs->super.init) {
                    if (rs->super.init != rd->super.init) {
                        /* User setting overrides existing init fn. Save it as a fallback */
                        FB_ALLOC_INSERT(&rd->super, fb, new, status, out);
                    }
                    new->super.init = rs->super.init;
                    new->super.team = rs->super.team;
                }
                ucc_list_insert_before(d, &new->super.list_elem);
                rd->start = rs->end;
                s         = s->next;
            } else {
                if (UCC_SCORE_INVALID != rs->super.score) {
                    rd->super.score = rs->super.score;
                }
                if (rs->super.init) {
                    if (rs->super.init != rd->super.init) {
                        /* User setting overrides existing init fn. Save it as a fallback */
                        FB_ALLOC_INSERT(&rd->super, fb, rd, status, out);
                    }
                    rd->super.init = rs->super.init;
                    rd->super.team = rs->super.team;
                }
                s = s->next;
                d = d->next;
            }
        }
    }
    while (s != src) {
        rs = ucc_container_of(s, ucc_msg_range_t, super.list_elem);
        if (rs->super.init) {
            new = MSG_RANGE_DUP(rs);
            if (new->super.score == UCC_SCORE_INVALID) {
                new->super.score = default_score;
            }
            ucc_list_add_tail(dest, &new->super.list_elem);
        }
        s = s->next;
    }
    /* remove potentially disabled ranges */
    ucc_list_for_each_safe(range, tmp, dest, super.list_elem) {
        if (0 == range->super.score) {
            ucc_list_del(&range->super.list_elem);
            ucc_msg_range_free(range);
        }
    }

    /* Merge consequtive ranges with the same score, same init fn,
       same team and same fallback sequence
       if any have been produced by the algorithm above */
    ucc_list_for_each_safe(range, tmp, dest, super.list_elem) { //NOLINT
        if (range->super.list_elem.next != dest) {
            next = ucc_container_of(range->super.list_elem.next,
                                    ucc_msg_range_t, super.list_elem);
            //NOLINTNEXTLINE
            if (range->super.score == next->super.score &&
                range->end == next->start &&
                range->super.init == next->super.init &&
                range->super.team == next->super.team &&
                1 == ucc_msg_range_fb_compare(range, next)) {
                next->start = range->start;
                ucc_list_del(&range->super.list_elem);
                ucc_msg_range_free(range);
            }
        }
    }
    return UCC_OK;
out:
    return status;
}

ucc_status_t ucc_coll_score_update(ucc_coll_score_t  *score,
                                   ucc_coll_score_t  *update,
                                   ucc_score_t        default_score,
                                   ucc_memory_type_t *mtypes,
                                   int                mt_n,
                                   uint64_t           colls)
{
    ucc_status_t      status;
    int               i, j;
    ucc_memory_type_t mt;

    if (mt_n == 0) {
        mt_n = UCC_MEMORY_TYPE_LAST;
    }

    for (i = 0; i < UCC_COLL_TYPE_NUM; i++) {
        if (!(colls & UCS_BIT(i))) {
            continue;
        }
        for (j = 0; j < mt_n; j++) {
            mt = (mtypes == NULL) ? (ucc_memory_type_t)j : mtypes[j];
            status = ucc_coll_score_update_one(
                &score->scores[i][mt],
                &update->scores[i][mt], default_score);
            if (UCC_OK != status) {
                return status;
            }
        }
    }
    return UCC_OK;
}

ucc_status_t
ucc_coll_score_update_from_str(const char *str,
                               const ucc_coll_score_team_info_t *info,
                               ucc_base_team_t *team,
                               ucc_coll_score_t *score)
{
    ucc_status_t      status;
    ucc_coll_score_t *score_str;

    status = ucc_coll_score_alloc_from_str(str, &score_str, info->size,
                                           info->init, team, info->alg_fn);
    if (UCC_OK != status) {
        return status;
    }

    status = ucc_coll_score_update(score, score_str,
                                   info->default_score,
                                   info->supported_mem_types,
                                   info->num_mem_types,
                                   info->supported_colls);
    ucc_coll_score_free(score_str);

    return status;
}

ucc_status_t ucc_coll_score_build_default(ucc_base_team_t        *team,
                                          ucc_score_t             default_score,
                                          ucc_base_coll_init_fn_t default_init,
                                          uint64_t                coll_types,
                                          ucc_memory_type_t      *mem_types,
                                          int mt_n, ucc_coll_score_t **score_p)
{
    ucc_coll_score_t *score;
    ucc_status_t      status;
    ucc_memory_type_t m;
    uint64_t          c;
    ucc_coll_type_t   ct;

    status = ucc_coll_score_alloc(&score);
    if (UCC_OK != status) {
        return status;
    }

    if (!mem_types) {
        mt_n = UCC_MEMORY_TYPE_LAST;
    }

    ucc_for_each_bit(c, coll_types) {
        for (m = UCC_MEMORY_TYPE_HOST; m < mt_n; m++) {
            ct = (ucc_coll_type_t)UCC_BIT(c);
            status = ucc_coll_score_add_range(
                score, ct, mem_types ? mem_types[m] : m, 0, UCC_MSG_MAX,
                default_score, default_init, team);
            if (UCC_OK != status) {
                ucc_coll_score_free(score);
                return status;
            }
        }
    }
    *score_p = score;
    return UCC_OK;
}

ucc_status_t ucc_coll_score_dup(const ucc_coll_score_t *in,
                                ucc_coll_score_t **     out)
{
    ucc_coll_score_t *score;
    ucc_status_t      status;
    int               i, j;

    status = ucc_coll_score_alloc(&score);
    if (UCC_OK != status) {
        return status;
    }
    for (i = 0; i < UCC_COLL_TYPE_NUM; i++) {
        for (j = 0; j < UCC_MEMORY_TYPE_LAST; j++) {
            status =
                ucc_score_list_dup(&in->scores[i][j], &score->scores[i][j]);
            if (UCC_OK != status) {
                return status;
            }
        }
    }
    *out = score;
    return status;
}

void ucc_coll_score_set(ucc_coll_score_t *score,
                        ucc_score_t       value)
{
    int               i, j;
    ucc_msg_range_t  *range;

    for (i = 0; i < UCC_COLL_TYPE_NUM; i++) {
        for (j = 0; j < UCC_MEMORY_TYPE_LAST; j++) {
            ucc_list_for_each(range, &score->scores[i][j], super.list_elem) {
                range->super.score = value;
            }
        }
    }
}
