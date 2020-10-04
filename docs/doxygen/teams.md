## Teams

The ucc\_team\_h is a team handle, which encapsulates the resources required for group operations such as collective communication operations. The participants of the group operations can either be an OS process, a control thread or a task. 

Create and destroy routines: ucc\_team\_create\_post routine is used to create the team handle and ucc\_team\_create\_test routine for learning the status of the create operation. The team handle is destroyed by the ucc\_team\_destroy operation. A team handle is customized using the user configured ucc\_team\_params\_t structure. 

\b Invocation semantics: The ucc\_team\_create\_post is a nonblocking collective operation, in which the participants are determined by the user-provided OOB collective operation. Overlapping of multiple ucc\_team\_create\_post operations are invalid. Posting a collective operation before the team handle is created is invalid. The team handle is destroyed by a blocking collective operation; the participants of this collective operation are the same as the create operation. When the user does not provide an OOB collective operation, all participants calling the ucc\_create\_post operation will be part of a new team created.

\b Communication Contexts: Each process or a thread participating in the team creation operation contributes one or more communication contexts to the operation. The number of contexts provided by all participants should be the same and each participant should provide the same type of context. The newly created team uses the context for collective operations. If the communication context abstracts the resources for the library, the collective operations on this team uses the resources provided by the context. 

\b Endpoints: That participants to the ucc\_team\_create\_post operation can provide an endpoint, a 64-bit unsigned integer. The endpoint is an address for communication. Each participant of the team has a unique integer as endpoint .i.e., the participants of the team do not share the same endpoint. For example, the user can bind the endpoint to the parallel programming model’s index such as OpenSHMEM PE, an OS process ID, or a thread ID. The UCC implementation can use the endpoint as an index to identify the resources required for communication such as communication contexts. When the user does not provide the endpoint, the library generates the endpoint, which can be queried by the user. In addition to the endpoint, the user can provide information about the endpoints such as whether the endpoint is a continuous range or not. 

\b Ordering: The collective operations on the team can either be ordered or unordered. In the ordered model, the UCC collectives are invoked in order .i.e., on a given team, each of the participants of the collective operation invokes the operation in the same order. In the unordered model, the collective operations are not necessarily invoked in the same order.

\b Interaction with Threads: The team can be created in either mode .i.e., the library initialized by UCC\_LIB\_THREAD\_MULTIPLE, UCC\_LIB\_THREAD\_SINGLE, or UCC\_LIB\_THREAD\_FUNNEDLED. In the UCC\_LIB\_THREAD\_MULTIPLE mode, each of the user threads can post a collective operation. However, it is not valid to post concurrent collectives operations from multiple threads to the same team. 

\b Memory per Team: A team can be configured by a memory descriptor described by ucc\_mem\_map\_params\_t structure. The memory can be used as an input and output buffers for the collective operation. This is particularly useful for PGAS programming models, where the input and output buffers are defined before the invocation operation. For example, the input and output buffers in the OpenSHMEM programming model are defined during the programming model initialization.

\b Synchronization Model: The team can be configured to support either synchronized collectives or non-synchronized collectives. If the UCC library is configured with synchronized collective operations and the team is configured with non-synchronized collective operations, the library might not be able to provide any optimizations and might support only synchronized collective operations.

\b Outstanding Calls: The user can configure maximum number of outstanding collective operations of any type for a given team.  This is represented by an unsigned integer.  This is provided as a hint to the library for resource management.

\b Team ID: The team identifier is a unique 64-bit unsigned integer for the given process .i.e, the team identifier should be unique for all teams it creates or participates. If the team identifier is provided by the user, it should be passed as a configuration parameter to the team create operation.


Split Team Operations

The team split routines provide an alternate way to create teams. All split routines require a parent team and all participants of the parent team call the split operation. The participants of the new team may include some or all participants of the parent team.

The newly created team shares the communication contexts with the parent team. The endpoint of the new team is contiguous and is not related to the parent team. It inherits the thread model, synchronization model, collective ordering model, outstanding collectives configuration, and memory descriptor from the parent team.

The split operation can be called by multiple threads, if the parent team to the split operations are different and if it agrees with the thread model of the UCC library.

Notes: The rationale behind requiring all participants of the parent team to participate in the split operation is to avoid overlapping participants between multiple split operations, which is known to increase the implementation complexity. Also, currently, higher-level programming models do not require these semantics.

