## Types of Collective Operations

A UCC collective operation is a group communication operation among the
participants of the team. All participants of the team are required to call the
collective operation. Each participant is represented by the endpoint that is
unique to the team used for the collective operation. This section provides a
set of routines for launching, progressing, and completing the collective
operations.

**Invocation semantics**: The ucc\_collective\_init routine is a non-blocking
collective operation to initialize the buffers, operation type, reduction type,
and other information required for the collective operation. All participants of
the team should call the initialize operation. The collective operation
is invoked using a ucc\_collective\_post operation.
ucc\_collective\_init\_and\_post operation initializes as well as post the
collective operation.

**Collective Type**: The collective operation supported by UCC is defined by the
enumeration ucc\_coll\_type\_t. The semantics are briefly described here,
however in most cases it agrees with the semantics of collective operations in
the popular programming models such as MPI and OpenSHMEM. When they differ, the
semantics changes are documented. All collective operations execute on the team.
For the collective operations defined by ucc\_coll\_type\_t, all participants of
the team are required to participate in the collective operations. Further the
team should be created with endpoints, where the “eps” should be ordered and
contiguous.

UCC supports three types of collective operations: (a) UCC\_{ALLTOALL, ALLTOALLV,
ALLGATHER, ALLGATHERV, ALLREDUCE, REDUCE_SCATTER, REDUCE_SCATTERV, BARRIER}
operations where all participants contribute to the results and receive the
results (b) UCC\_{REDUCE, GATHER, GATHERV, FANIN} where all participants
contribute to the result and one participant receives the result. The
participant receiving the result is designated as root. (c) UCC\_{BROADCAST,
SCATTER, SCATTERV, FANOUT} where one participant contributes to the result, and
all participants receive the result. The participant contributing to the result
is designated as root.

+ The UCC\_COLL\_TYPE\_BCAST operation moves the data from the root participant
to all participants in the team.

+ The UCC\_COLL\_TYPE\_BARRIER synchronizes all participants of the collective
operation. In this routine, first, each participant waits for all other
participants to enter the operation. Then, once it learns the entry of all other
participants into the operation, it exits the operation completing it locally.

+ In the UCC\_COLL\_TYPE\_FAN\_IN operation, the root participant synchronizes
with all participants of the team. The non-root completes when it sends
synchronizing message to the root. Unlike UCC\_COLL\_TYPE\_BARRIER, it doesn’t
have to synchronize with the rest of the non-root participants. The root
participant completes the operation when it receives synchronizing messages from
all non-root participants of the team.


+ The UCC\_COLL\_TYPE\_FAN\_OUT operation is a synchronizing operation like
UCC\_COLL\_TYPE\_FAN\_OUT. In this operation, the root participant sends a
synchronizing message to all non-root participants and completes. The non-root
participant completes once it receives a message from the root participant.


+ In the UCC\_COLL\_TYPE\_GATHER operation, each participant of the collective
operation sends data to the root participant. All participants send the same
amount of data (block\_size) to the root. The size of the block is
“dt\_elem\_size * count”. The total amount of data received by the root is equal
to block_size * num\_participants. Here, the “count” represents the number of
data elements. The
"dt_elem_size" represents the size of the data element in bytes. The
"num_participants" represents the number of participants in the team. The data on
the root is placed in the receive buffer ordered by the “ep” ordering. For
example, if the participants’ endpoints are ordered as “ep\_a” to “ep\_n”, the
data from the participant with ep_i is placed as an “ith” block on the receive
buffer.

+ The UCC\_COLL\_TYPE\_ALLGATHER operation is similar to UCC\_COLL\_TYPE\_GATHER
with one exception. Unlike in GATHER operation, the result is available at all
participants’ receive buffer instead of only at the root participant.

    Each participant sends the data of size "block_size" to all other participants
in the collective operation. The size of the block is “dt\_elem\_size * count”.
Here, the “count” represents the number of data elements. The "dt_elem_size"
represents the size of the data element in bytes. The data on each participant
is placed in the receive buffer ordered by the “ep” ordering. For example, if
the participants’ endpoints are ordered as “ep\_a” to “ep\_n”, the data from the
participant with ep_i is placed as an “ith” block on the receive buffer.

+ In the UCC\_COLL\_TYPE\_SCATTER operation, the root participant of the
collective operation sends data to all other participants. It sends the same
amount of data (block_size) to all participants. The size of the block
(block_size) is “dt_elem_size * count”. The total amount of data sent by the
root is equal to block_size * num\_participants. Here, the “count” represents the
number of data elements. The "dt_elem_size" represents the size of the data
element in bytes. The "num_participants" represents the number of participants in
the team.

+ In the UCC\_COLL\_TYPE\_ALLTOALL collective operation, all participants
exchange a fixed amount of the data. For a given participant, the size of data
in src buffer is “size”, where size is dt\_elem\_size * count * num_participants.
Here, the “count” represents the number of data elements per destination. The "dt_elem_size"
represents the size of the data element in bytes. The "num_participants" represents
the number of participants in the team. The size of src buffer is the same as
the dest buffer, and it is the same across all participants. Each participant
exchanges “dt\_elem\_size * count “ data with every participant of the collective.

+ In UCC\_COLL\_TYPE\_REDUCE collective the element-wise reduction operation is
performed on the src buffer of all participants in the collective operation. The
result is stored on the dst buffer of the root. The size of src buffer and dst
buffer is the same, which is equal to “dt_elem_size * count”. Here, the “count”
represents the number of data elements. The "dt_elem_size" represents the size
of the data element in bytes.

+ The UCC\_COLL\_TYPE\_ALLREDUCE first performs an element-wise reduction on the
src buffers of all participants. Then the result is distributed to all
participants. After the operation, the results are available on the dst buffer
of all participants. The size of src buffer and dst buffer is the same for all
participants. The size of src buffer and dst buffer is the same, which is equal
to “dt_elem_size * count”. Here, the “count” represents the number of data
elements. The "dt_elem_size" represents the size of the data element in bytes.

+ The UCC\_COLL\_TYPE\_REDUCE\_SCATTER first performs an element-wise reduction
on the src buffer and then scatters the result to the dst buffer. The "size" of
src buffer is “count * dt_elem_size”, where dt_elem_size is the number of bytes
for the data type element and count is the number of elements of that datatype.
It is the user’s responsibility to ensure that data and the result are
equally divisible among the participants. Assuming that the result is
divided into “n” blocks, the ith block is placed in the receive buffer
of endpoint “i”. Like other collectives, for this collective, the “ep”
should be ordered and contiguous.


**INPLACE**: When INPLACE is set for UCC\_COLL\_TYPE\_REDUCE\_SCATTER,
UCC\_COLL\_TYPE\_REDUCE, UCC\_COLL\_TYPE\_ALLREDUCE, UCC\_COLL\_TYPE\_SCATTER,
and UCC\_COLL\_TYPE\_ALLTOALL the receive buffers act as both send and receive
buffer.

For UCC\_COLL\_TYPE\_BCAST operation, setting INPLACE flag has no impact.

**The "v" Variant Collective Types**: The UCC\_COLL\_TYPE\_{ALLTOALLV, SCATTERV,
GATHERV, and REDUCE\_SCATTERV} operations add flexibility to their counter
parts (.i.e., ALLTOALL, SCATTER, GATHER, and REDUCE\_SCATTER) in that the
location of data for the send and receive are specified by displacement arrays.

**Reduction Types**: The reduction operation supported by UCC\_{ALLREDUCE,
REDUCE, REDUCE\_SCATTER, REDUCE\_SCATTERV} operation is defined by the
enumeration ucc\_reduction\_op\_t. The valid datatypes
for the reduction is defined by the enumeration ucc\_datatype\_t.

\b Ordering: The team can be configured for ordered collective operations or
unordered collective operations. For unordered collectives, the user is required
to provide the “tag”, which is an unsigned 64-bit integer.

\b Synchronized and Non-Synchronized Collectives: In the synchronized collective
model, on entry, the participants cannot read or write to other participants
without ensuring all participants have entered the collective operation. On the
exit of the collective operation, the participants may exit after all
participants have completed the reading or writing to the buffers.

In the non-synchronized collective model, on entry, the participants can read or
write to other participants. If the input and output buffers are defined on the
team and RMA operations are used for data transfer, it is the responsibility of
the user to ensure the readiness of the buffer. On exit, the participants may
exit once the read and write to the local buffers are completed.

\b Buffer Ownership: The ownership of input and output buffers are transferred
from the user to the library after invoking the ucc\_collective\_init routine.
On return from the routine, the ownership is transferred back to the user on
ucc\_collective\_finalize.
However, after invoking and returning from ucc\_collective\_post or
ucc\_collective\_init\_and\_post routines, the ownership stays with the library
and it is returned to the user, when the collective is completed.

\b The table below lists the necessary fields that user must initialize
depending on the collective operation type.
\image latex ucc\_coll\_args\_table1.pdf width=\textwidth
\image latex ucc\_coll\_args\_table2.pdf width=\textwidth
