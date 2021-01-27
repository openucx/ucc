/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef RECURSIVE_KNOMIAL_H_
#define RECURSIVE_KNOMIAL_H_

enum {
    KN_NODE_BASE,  /*< Participates in the main loop of the recursive KN algorithm */
    KN_NODE_PROXY, /*< Participates in the main loop and receives/sends the data from/to EXTRA */
    KN_NODE_EXTRA  /*< Just sends/receives the data to/from its PROXY */
};

/**
 *  @param [in]  _size           Size of the team
 *  @param [in]  _radix          Knomial radix
 *  @param [out] _pow_radix_sup  Smallest integer such that _radix**_pow_radix_sup >= _size
 *  #param [out] _full_pow_size  Largest power of _radix <= size. Ie. it is equal to
                                 _radix**_pow_radix_sup if _radix**_pow_radix_sup == size, OR
                                 _radix**(_pow_radix_sup - 1) otherwise
*/
#define CALC_POW_RADIX_SUP(_size, _radix, _pow_radix_sup, _full_pow_size) do{ \
        int pk = 1;                                                           \
        int fs = _radix;                                                      \
        while (fs < _size) {                                                  \
            pk++; fs*=_radix;                                                 \
        }                                                                     \
        _pow_radix_sup = pk;                                                  \
        _full_pow_size = (fs != _size) ? fs/_radix : fs;                      \
    }while(0)

/**
 *  Initializes recursive knomial tree attributes.
 *  @param [in]  _radix            Knomial radix
 *  @param [in]  _myrank           Rank in a team
 *  @param [in]  _size             Team size
 *  @param [out] _pow_radix_super  (see above)
 *  @param [out] _full_pow_size    (see above)
 *  @param [out] _n_full_subtrees  Number of full knomial subtrees that fit into team size
 *  @param [out] _full_size        Total number of nodes in the set of full subtrees
 *  @param [out] _node_type        Rank node type
 */
#define KN_RECURSIVE_SETUP(__radix, __myrank, __size, __pow_k_sup,         \
                           __full_pow_size, __n_full_subtrees,             \
                           __full_size, __node_type) do{                   \
        CALC_POW_RADIX_SUP(__size, __radix, __pow_k_sup, __full_pow_size); \
        __n_full_subtrees = __size / __full_pow_size;                      \
        __full_size = __n_full_subtrees*__full_pow_size;                   \
        __node_type = __myrank >= __full_size ? KN_NODE_EXTRA :            \
            (__size > __full_size && __myrank < __size - __full_size ?     \
             KN_NODE_PROXY : KN_NODE_BASE);                                \
    }while(0)

#define KN_RECURSIVE_GET_PROXY(__myrank, __full_size) (__myrank - __full_size)
#define KN_RECURSIVE_GET_EXTRA(__myrank, __full_size) (__myrank + __full_size)

#endif
