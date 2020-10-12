### Check style with clang-format

####  Usage 
clang-format --style=file ucc.h > ucc.formatted.h

### Examples where format with clang-format fails

clang-format tool formats the C source. Though the .clang-format style captures
the preferred style by the UCC project, there are some edge cases where it
fails. The examples below captures the scenarios: 


```C
Good
typedef void(*ucc_reduction_dtype_mapper_t)(void *invec, void *inoutvec,
                                           ucc_count_t *count, ucc_datatype_t dtype);
```                                            
                                            
clang-format generated

```C
typedef void (*ucc_reduction_dtype_mapper_t)(void *invec, void *inoutvec,
                                             ucc_count_t *  count,
                                             ucc_datatype_t dtype);
```
                                             
                                             
Good

```C
typedef struct ucc_context_oob_coll {
    int                  (*allgather)(void *src_buf, void *recv_buf, size_t size,
                                      void *allgather_info,  void **request);
    ucc_status_t         (*req_test)(void *request);
    ucc_status_t         (*req_free)(void *request);
    uint32_t             participants;
    void                 *coll_info;
}  ucc_context_oob_coll_t;                                   
```
                                             
clang-format generated

```C
typedef struct ucc_context_oob_coll {
    int (*allgather)(void *src_buf, void *recv_buf, size_t size,
                     void *allgather_info, void **request);
    ucc_status_t (*req_test)(void *request);
    ucc_status_t (*req_free)(void *request);
    uint32_t participants;
    void *   coll_info;
} ucc_context_oob_coll_t;
```
