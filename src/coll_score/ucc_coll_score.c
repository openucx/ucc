/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "ucc_coll_score.h"
#include "utils/ucc_string.h"
#include "utils/ucc_coll_utils.h"

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
    r->start   = start;
    r->end     = end;
    r->score   = msg_score;
    r->init    = init;
    r->team    = team;
    list       = &score->scores[ucc_ilog2(coll_type)][mem_type];
    insert_pos = list;
    ucc_list_for_each(range, list, list_elem) {
        if (start >= range->end) {
            insert_pos = &range->list_elem;
        } else {
            break;
        }
    }
    ucc_list_insert_after(insert_pos, &r->list_elem);
    /*sanity check: ranges shuold not overlap */
    next = r->list_elem.next;
    if ((next != list) &&
        (ucc_container_of(next, ucc_msg_range_t, list_elem)->start < r->end)) {
        ucc_error("attempt to add overlaping range");
        ucc_list_del(&r->list_elem);
        ucc_free(r);
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
            ucc_list_destruct(&score->scores[i][j], ucc_msg_range_t, ucc_free,
                              list_elem);
        }
    }
}

static ucc_status_t ucc_score_list_dup(ucc_list_link_t *src,
                                       ucc_list_link_t *dst)
{
    ucc_msg_range_t *range, *r;
    ucc_list_head_init(dst);
    /* progress registered progress fns */
    ucc_list_for_each(range, src, list_elem) {
        r = ucc_malloc(sizeof(*r), "ucc_msga_range");
        if (!r) {
            ucc_error("failed to allocate %zd bytes for ucc_msg_range",
                      sizeof(*r));
            goto error;
        }
        memcpy(r, range, sizeof(*r));
        ucc_list_add_tail(dst, &r->list_elem);
    }
    return UCC_OK;
error:
    ucc_list_for_each_safe(range, r, dst, list_elem) {
        ucc_list_del(&range->list_elem);
        ucc_free(range);
    }
    ucc_assert(ucc_list_is_empty(dst));
    return UCC_ERR_NO_MEMORY;
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
        goto destruct;
    }

    while (!(ucc_list_is_empty(&lst1) && ucc_list_is_empty(&lst2))) {
        if (ucc_list_is_empty(&lst1)) {
            ucc_list_add_tail(out, &(ucc_list_extract_head(&lst2,
                                     ucc_msg_range_t, list_elem)->list_elem));
            continue;
        }
        if (ucc_list_is_empty(&lst2)) {
            ucc_list_add_tail(out, &(ucc_list_extract_head(&lst1,
                                     ucc_msg_range_t, list_elem)->list_elem));
            continue;
        }
        r1 = ucc_list_head(&lst1, ucc_msg_range_t, list_elem);
        r2 = ucc_list_head(&lst2, ucc_msg_range_t, list_elem);
        left = (r1->start < r2->start) ? r1 : r2; //NOLINT

        if (r1->start == r2->start) {
            if (r1->end == r2->end) {
                best = (r1->score > r2->score) ? r1 : r2;
                left                  = (best == r1) ? r2 : r1;
                ucc_list_del(&r1->list_elem);
                ucc_list_del(&r2->list_elem);
                ucc_list_add_tail(out, &best->list_elem);
                ucc_free(left);
                continue;
            }
            left = (r1->end < r2->end) ? r1 : r2;
        }
        right = (left == r1) ? r2 : r1;
        if (left->end <= right->start) {
            /* ranges don't overlap - copy over */
            ucc_list_del(&left->list_elem);
            ucc_list_add_tail(out, &left->list_elem);
        } else if (left->end < right->end) {
            if (left->score >= right->score) {
                right->start = left->end;
                ucc_list_del(&left->list_elem);
                ucc_list_add_tail(out, &left->list_elem);
            } else {
                ucc_list_del(&left->list_elem);
                if (right->start > left->start) {
                    /* non zero overlap leftover */
                    left->end = right->start;
                    ucc_list_add_tail(out, &left->list_elem);
                } else {
                    ucc_free(left);
                }
            }
        } else {
            if (left->score >= right->score) {
                /* just drop inner range with lower score */
                ucc_list_del(&right->list_elem);
                ucc_free(right);
            } else {
                new = ucc_malloc(sizeof(*new), "ucc_msg_range");
                if (!new) {
                    ucc_error("failed to allocate %zd bytes for ucc_msg_range",
                              sizeof(*new));
                    status = UCC_ERR_NO_MEMORY;
                    goto destruct;
                }
                memcpy(new, left, sizeof(*new));
                new->end = right->start;
                ucc_list_add_tail(out, &new->list_elem);
                ucc_list_del(&right->list_elem);
                ucc_list_add_tail(out, &right->list_elem);
                left->start = right->end;
            }
        }
    }
    /* Merge consequtive ranges with the same score and same init fn
       if any have been produced by the algorithm above */
    ucc_list_for_each_safe(range, temp, out, list_elem) {
        if (range->list_elem.next != out) {
            next = ucc_container_of(range->list_elem.next, ucc_msg_range_t,
                                    list_elem);
            if (range->score == next->score && range->end == next->start &&
                range->init == next->init && range->team == next->team) {
                next->start = range->start;
                ucc_list_del(&range->list_elem);
                ucc_free(range);
            }
        }
    }
    return UCC_OK;
destruct:
    ucc_list_destruct(&lst2, ucc_msg_range_t, ucc_free, list_elem);
    ucc_list_destruct(&lst1, ucc_msg_range_t, ucc_free, list_elem);
    ucc_list_destruct(out, ucc_msg_range_t, ucc_free, list_elem);
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

static ucc_status_t str_to_mem_type(const char *str, unsigned *mt_n,
                                    ucc_memory_type_t **mt)
{
    ucc_status_t      status = UCC_OK;
    char **           tokens;
    unsigned          i, n_tokens;
    ucc_memory_type_t t;
    tokens = ucc_str_split(str, ",");
    if (!tokens) {
        status = UCC_ERR_INVALID_PARAM;
        goto out;
    }
    n_tokens = ucc_str_split_count(tokens);
    *mt = ucc_malloc(n_tokens * sizeof(ucc_memory_type_t), "ucc_mem_types");
    if (!(*mt)) {
        ucc_error("failed to allocate %zd bytes for ucc_mem_types",
                  sizeof(ucc_memory_type_t) * n_tokens);
        status = UCC_ERR_NO_MEMORY;
        goto out;
    }
    *mt_n = 0;
    for (i = 0; i < n_tokens; i++) {
        t = ucc_mem_type_from_str(tokens[i]);
        if (t == UCC_MEMORY_TYPE_LAST) {
            /* entry does not match any memory type name */
            ucc_free(*mt);
            *mt    = NULL;
            status = UCC_ERR_NOT_FOUND;
            goto out;
        }
        (*mt)[*mt_n] = t;
        (*mt_n)++;
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
            continue;
        }
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
                                             ucc_rank_t team_size, //NOLINT
                                             ucc_base_coll_init_fn_t init,
                                             ucc_base_team_t *team,
                                             ucc_alg_id_to_init_fn_t alg_fn)
{
    ucc_status_t            status   = UCC_OK;
    ucc_coll_type_t        *ct       = NULL;
    ucc_memory_type_t      *mt       = NULL;
    size_t                 *msg      = NULL;
    ucc_rank_t             *tsizes   = NULL;
    ucc_base_coll_init_fn_t alg_init = NULL;
    const char*             alg_id   = NULL;
    ucc_score_t             score_v  = UCC_SCORE_INVALID;
    int                     ts_skip  = 0;
    char                  **tokens;
    unsigned i, n_tokens, ct_n, mt_n, c, m, n_ranges, r, n_tsizes;

    mt_n = ct_n = n_ranges = n_tsizes = 0;
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
        if (!mt && UCC_OK == str_to_mem_type(tokens[i], &mt_n, &mt)) {
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

        //TODO add parsing of msg ranges and team size ranges
        goto out;
    }
    if (tsizes) {
        /* Team size qualifier was provided: check if we should apply this
           str setting to the  current team */
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
        if (!ct)
            ct_n = UCC_COLL_TYPE_NUM;
        if (!mt)
            mt_n = UCC_MEMORY_TYPE_LAST;
        if (!msg)
            n_ranges = 1;
        for (c = 0; c < ct_n; c++) {
            for (m = 0; m < mt_n; m++) {
                ucc_coll_type_t   coll_type = ct ? ct[c] : UCC_BIT(c);
                ucc_memory_type_t mem_type  = mt ? mt[m] : m;
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
    ucc_free(mt);
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
    ucc_error("failed to parse UCC_*_SCORE parameter: %s", tokens[i]);
error:
    *score_p = NULL;
    ucc_coll_score_free(score);
    ucc_str_split_free(tokens);
    return status;
}

#define MSG_RANGE_DUP(_src)                                                    \
    ({                                                                         \
        ucc_msg_range_t *_dup = ucc_malloc(sizeof(*_dup), "ucc_msg_range");    \
        if (!_dup) {                                                           \
            ucc_error("failed to allocate %zd bytes for ucc_msg_range",        \
                      sizeof(*_dup));                                          \
            status = UCC_ERR_NO_MEMORY;                                        \
            goto out;                                                          \
        }                                                                      \
        memcpy(_dup, _src, sizeof(*_dup));                                     \
        _dup;                                                                  \
    })

static ucc_status_t ucc_coll_score_update_one(ucc_list_link_t *dest,
                                              ucc_list_link_t *src,
                                              ucc_score_t      default_score)
{
    ucc_list_link_t *s = src->next;
    ucc_list_link_t *d = dest->next;
    ucc_msg_range_t *range, *tmp, *next, *rs, *rd, *new;
    ucc_status_t status;
    if (ucc_list_is_empty(src) || ucc_list_is_empty(dest)) {
        return UCC_OK;
    }
    while (s != src && d != dest) {
        rs = ucc_container_of(s, ucc_msg_range_t, list_elem);
        rd = ucc_container_of(d, ucc_msg_range_t, list_elem);
        ucc_assert((NULL == rs->init) || (NULL != rs->team));
        if (rd->start >= rs->end) {
            /* skip src range - no overlap */
            s = s->next;
            if (rs->init) {
                new = MSG_RANGE_DUP(rs);
                if (new->score == UCC_SCORE_INVALID) {
                    new->score = default_score;
                }
                ucc_list_insert_before(d, &new->list_elem);
            }
        } else if (rd->end <= rs->start) {
            /* no overlap - inverse case: skip dst range */
            d = d->next;
        } else if (rd->start <  rs->start) {
            new       = MSG_RANGE_DUP(rd);
            new->end  = rs->start;
            rd->start = rs->start;
            ucc_list_insert_before(d, &new->list_elem);
        } else if (rd->start > rs->start) {
            if (rs->init) {
                new = MSG_RANGE_DUP(rs);
                if (new->score == UCC_SCORE_INVALID) {
                    new->score = default_score;
                }
                new->end = rd->start;
                ucc_list_insert_before(d, &new->list_elem);
            }
            rs->start = rd->start;
        } else {
            /* same start */
            if (rs->end > rd->end) {
                if (UCC_SCORE_INVALID != rs->score) {
                    rd->score = rs->score;
                }
                if (rs->init) {
                    rd->init = rs->init;
                    rd->team = rs->team;
                }
                rs->start = rd->end;
                d         = d->next;
            } else if (rs->end < rd->end) {
                new      = MSG_RANGE_DUP(rd);
                new->end = rs->end;
                if (UCC_SCORE_INVALID != rs->score) {
                    new->score = rs->score;
                }
                if (rs->init) {
                    new->init = rs->init;
                    new->team = rs->team;
                }
                ucc_list_insert_before(d, &new->list_elem);
                rd->start = rs->end;
                s         = s->next;
            } else {
                if (UCC_SCORE_INVALID != rs->score) {
                    rd->score = rs->score;
                }
                if (rs->init) {
                    rd->init = rs->init;
                    rd->team = rs->team;
                }
                s = s->next;
                d = d->next;
            }
        }
    }
    while (s != src) {
        rs = ucc_container_of(s, ucc_msg_range_t, list_elem);
        if (rs->init) {
            new = MSG_RANGE_DUP(rs);
            ucc_list_add_tail(dest, &new->list_elem);
        }
        s = s->next;
    }
    /* remove potentially disabled ranges */
    ucc_list_for_each_safe(range, tmp, dest, list_elem) {
        if (0 == range->score) {
            ucc_list_del(&range->list_elem);
            ucc_free(range);
        }
    }

    /* Merge consequtive ranges with the same score and same init fn
       if any have been produced by the algorithm above */
    ucc_list_for_each_safe(range, tmp, dest, list_elem) { //NOLINT
        if (range->list_elem.next != dest) {
            next = ucc_container_of(range->list_elem.next, ucc_msg_range_t,
                                    list_elem);
            //NOLINTNEXTLINE
            if (range->score == next->score && range->end == next->start &&
                range->init == next->init && range->team == next->team) {
                next->start = range->start;
                ucc_list_del(&range->list_elem);
                ucc_free(range);
            }
        }
    }
    return UCC_OK;
out:
    return status;
}

ucc_status_t ucc_coll_score_update(ucc_coll_score_t *score,
                                   ucc_coll_score_t *update,
                                   ucc_score_t       default_score)
{
    ucc_status_t      status;
    int               i, j;
    for (i = 0; i < UCC_COLL_TYPE_NUM; i++) {
        for (j = 0; j < UCC_MEMORY_TYPE_LAST; j++) {
            status = ucc_coll_score_update_one(
                &score->scores[i][j], &update->scores[i][j], default_score);
            if (UCC_OK != status) {
                return status;
            }
        }
    }
    return UCC_OK;
}

ucc_status_t ucc_coll_score_update_from_str(const char *            str,
                                            ucc_coll_score_t       *score,
                                            ucc_rank_t              team_size,
                                            ucc_base_coll_init_fn_t init,
                                            ucc_base_team_t        *team,
                                            ucc_score_t             def_score,
                                            ucc_alg_id_to_init_fn_t alg_fn)
{
    ucc_status_t      status;
    ucc_coll_score_t *score_str;
    status = ucc_coll_score_alloc_from_str(str, &score_str, team_size, init,
                                           team, alg_fn);
    if (UCC_OK != status) {
        return status;
    }
    status = ucc_coll_score_update(score, score_str, def_score);
    ucc_coll_score_free(score_str);
    return status;
}

ucc_status_t ucc_coll_score_build_default(ucc_base_team_t        *team,
                                          ucc_score_t             default_score,
                                          ucc_base_coll_init_fn_t default_init,
                                          ucc_coll_type_t         coll_types,
                                          ucc_memory_type_t      *mem_types,
                                          int mt_n, ucc_coll_score_t **score_p)
{
    ucc_coll_score_t *score;
    ucc_status_t      status;
    ucc_memory_type_t m;
    uint64_t          c;
    status = ucc_coll_score_alloc(&score);
    if (UCC_OK != status) {
        return status;
    }

    if (!mem_types) {
        mt_n = UCC_MEMORY_TYPE_LAST;
    }

    ucc_for_each_bit(c, coll_types) {
        for (m = 0; m < mt_n; m++) {
            status = ucc_coll_score_add_range(
                score, UCC_BIT(c), mem_types ? mem_types[m] : m, 0, UCC_MSG_MAX,
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
