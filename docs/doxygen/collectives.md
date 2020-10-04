## Starting and Completing the Collectives

A UCC collective operation is a group communication operation among the participants of the team. All participants of the team are required to call the collective operation. Each participant is represented by the endpoint that is unique to the team used for the collective operation. This section provides a set of routines for launching, progressing, and completing the collective operations. 

\b Invocation semantics: The ucc\_collective\_init routine is a non-blocking collective operation to initialize the buffers, operation type, reduction type, and other information required for the collective operation. All participants of the team should call the initialize operation. The routine returns once the participants enter the collective initialize operation. The collective operation is invoked using a ucc\_collective\_post operation. ucc\_collective\_init\_and\_post operation initializes as well as post the collective operation.

\b Collective type: The collective operation supported by UCC is defined by the enumeration ucc\_coll\_type\_t. It supports three types of collective operations: (a) UCC\_{ALLTOALL, ALLGATHER, ALLREDUCE} operations where all participants contribute to the results and receive the results (b) UCC\_{REDUCE, GATHER, FANIN} where all participants contribute to the result and one participant receives the result. The participant receiving the result is designated as root. (c) UCC\_{BROADCAST, MULTICAST, SCATTER, FANOUT} where one participant contributes to the result, and all participants receive the result. The participant contributing to the result is designated as root.

\b Reduction types: The reduction operation supported by UCC\_{ALLREDUCE, REDUCE} operation is defined by the enumeration ucc\_op\_t. The valid datatypes for the reduction is defined by the enumeration ucc\_datatype\_t.

\b Ordering: The team can be configured for ordered collective operations or unordered collective operations. For unordered collectives, the user is required to provide the “tag”, which is an unsigned 64-bit integer. 

\b Synchronized and Non-Synchronized Collectives: In the synchronized collective model, on entry, the participants cannot read or write to other participants without ensuring all participants have entered the collective operation. On the exit of the collective operation, the participants may exit after all participants have completed the reading or writing to the buffers.

In the non-synchronized collective model, on entry, the participants can read or write to other participants. If the input and output buffers are defined on the team and RMA operations are used for data transfer, it is the responsibility of the user to ensure the readiness of the buffer. On exit, the participants may exit once the read and write to the local buffers are completed. 

\b Buffer Ownership: The ownership of input and output buffers are transferred from the user to the library after invoking the ucc\_collective\_init routine and on return from the routine, the ownership is transferred back to the user. However, after invoking and returning from ucc\_collective\_post or ucc\_collective\_init\_and\_post routines, the ownership stays with the library and it is returned to the user, when the collective is completed. 

