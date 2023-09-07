/**
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef TWO_TREE_H_
#define TWO_TREE_H_

enum {
    LEFT_CHILD,
    RIGHT_CHILD
};

typedef struct ucc_dbt_single_tree {
   ucc_rank_t rank;
   ucc_rank_t size;
   ucc_rank_t root;
   ucc_rank_t parent;
   ucc_rank_t children[2];
   int        height;
   int        recv;
} ucc_dbt_single_tree_t;

static inline ucc_rank_t get_root(ucc_rank_t size)
{
    ucc_rank_t r = 1;

    while (r <= size) {
        r *= 2;
    }
    return r/2 - 1;
}

static inline int get_height(ucc_rank_t rank)
{
    int h = 1;

    if (rank % 2 == 0) {
        return 0;
    }

    rank++;
    while ((rank & (1 << h)) == 0) {
        h++;
    }
    return h;
}

static inline ucc_rank_t get_left_child(ucc_rank_t rank, int height)
{
    ucc_rank_t sub_height;

    if (height == 0) {
        return -1;
    }

    sub_height = 1 << (height - 1);
    return rank - sub_height;
}

static inline ucc_rank_t get_right_child(ucc_rank_t size, ucc_rank_t rank,
                                         int height, ucc_rank_t root)
{
    ucc_rank_t sub_right_root, sub_height;

    if (rank == size - 1 || height == 0) {
        return -1;
    }

    sub_right_root = get_root(size - rank - 1) + 1;
    sub_height     = 1 << (height - 1);

    if (rank == root) {
        return rank + sub_right_root;
    }
    return (rank + sub_height < size) ? rank + sub_height
                                      : rank + sub_right_root;
}

static inline void get_children(ucc_rank_t size, ucc_rank_t rank, int height,
                                ucc_rank_t root, ucc_rank_t *l_c,
                                ucc_rank_t *r_c)
{
    *l_c = get_left_child(rank, height);
    *r_c = get_right_child(size, rank, height, root);
}

static inline int get_parent(int vsize, int vrank, int height, int troot)
{
    if (vrank == troot) {
        return -1;
    } else if (height == 0) {
        return ((((vrank/2) % 2 == 0) && (vrank + 1 != vsize))) ? vrank + 1
                                                                : vrank - 1;
    } else {
        vrank++;
        if ((((1<<(height+1)) & vrank) > 0) || (vrank + (1<<height)) > vsize) {
            return vrank - (1<<height) - 1;
        } else {
            return vrank + (1<<height) - 1;
        }
    }
}

static inline void ucc_two_tree_build_t2_mirror(ucc_dbt_single_tree_t t1,
                                 ucc_dbt_single_tree_t *t2)
{
    ucc_rank_t            size = t1.size;
    ucc_dbt_single_tree_t t;

    t.size                  = size;
    t.height                = t1.height;
    t.rank                  = size - 1 - t1.rank;
    t.root                  = size - 1 - t1.root;
    t.parent                = (t1.parent == -1) ? -1 : size - 1 - t1.parent;
    t.children[LEFT_CHILD]  = (t1.children[RIGHT_CHILD] == -1) ? -1 :
                               size - 1 - t1.children[RIGHT_CHILD];
    t.children[RIGHT_CHILD] = (t1.children[LEFT_CHILD] == -1) ? -1 :
                               size - 1 - t1.children[LEFT_CHILD];
    t.recv                  = 0;

    *t2 = t;
}

static inline void ucc_two_tree_build_t2_shift(ucc_dbt_single_tree_t t1,
                                               ucc_dbt_single_tree_t *t2)
{
    ucc_rank_t            size = t1.size;
    ucc_dbt_single_tree_t t;

    t.size                  = size;
    t.height                = t1.height;
    t.rank                  = (t1.rank + 1) % size;
    t.root                  = (t1.root + 1) % size;
    t.parent                = (t1.parent == -1) ? -1 : (t1.parent + 1) % size;
    t.children[LEFT_CHILD]  = (t1.children[LEFT_CHILD] == -1) ? -1 :
                              (t1.children[LEFT_CHILD] + 1) % size;
    t.children[RIGHT_CHILD] = (t1.children[RIGHT_CHILD] == -1) ? -1 :
                              (t1.children[RIGHT_CHILD] + 1) % size;
    t.recv                  = 0;

    *t2 = t;
}

static inline void ucc_two_tree_build_t1(ucc_rank_t rank, ucc_rank_t size,
                                         ucc_dbt_single_tree_t *t1)
{
    int         height   = get_height(rank);
    ucc_rank_t  root     = get_root(size);
    ucc_rank_t  parent   = get_parent(size, rank, height, root);

    get_children(size, rank, height, root, &t1->children[LEFT_CHILD],
                 &t1->children[RIGHT_CHILD]);
    t1->height = height;
    t1->parent = parent;
    t1->size   = size;
    t1->rank   = rank;
    t1->root   = root;
    t1->recv   = 0;
}

static inline ucc_rank_t ucc_two_tree_convert_rank_for_shift(ucc_rank_t rank,
                                                             ucc_rank_t size)
{
    ucc_rank_t i;
    for (i = 0; i < size; i++) {
        if (rank == (i + 1) % size) {
            break;
        }
    }
    return i;
}

static inline ucc_rank_t ucc_two_tree_convert_rank_for_mirror(ucc_rank_t rank,
                                                              ucc_rank_t size)
{
    ucc_rank_t i;
    for (i = 0; i < size; i++) {
        if (rank == size - 1 - i) {
            break;
        }
    }
    return i;
}

static inline void ucc_two_tree_build_t2(ucc_rank_t rank, ucc_rank_t size,
                                         ucc_dbt_single_tree_t *t2) {
    ucc_rank_t temp_rank = (size % 2) ?
        ucc_two_tree_convert_rank_for_shift(rank, size) :
        ucc_two_tree_convert_rank_for_mirror(rank, size);
    ucc_dbt_single_tree_t t1_temp;

    ucc_two_tree_build_t1(temp_rank, size, &t1_temp);
    if (size % 2) {
        ucc_two_tree_build_t2_shift(t1_temp, t2);
    } else {
        ucc_two_tree_build_t2_mirror(t1_temp, t2);
    }
}

static inline void ucc_two_tree_build_trees(ucc_rank_t rank, ucc_rank_t size,
                                            ucc_dbt_single_tree_t *t1,
                                            ucc_dbt_single_tree_t *t2)
{
    ucc_two_tree_build_t1(rank, size, t1);
    ucc_two_tree_build_t2(rank, size, t2);
}

#endif
