## Communication Context


The ucc\_context\_h is a communication context handle. It can encapsulate resources required for collective operations on team handles. The contexts are created by the ucc\_context\_create operation and destroyed by the ucc\_context\_destroy operation. The create operation takes in user-configured ucc\_context\_params\_t structure to customize the context handle. The attributes of the context created can be queried using the ucc\_context\_get\_attribs operation. 

When no out-of-band operation (OOB) is provided, the ucc\_context\_create operation is local requiring no communication with other participants. When OOB operation is provided, all participants of the OOB operation should participate in the create operation. If the context operation is a collective operation, the ucc\_context\_destroy operation is also a collective operation .i.e., all participants should call the destroy operation.

The context can be created as an exclusive type or shared type by passing constants UCC\_CONTEXT\_EXCLUSIVE and UCC\_CONTEXT\_SHARED respectively to the ucc\_context\_params\_t structure. When context is created as a shared type, the same context handle can be used to create multiple teams. When context is created as an exclusive type, the context can be used to create multiple teams but the team handles cannot be valid at the same time; a valid team is defined as a team object where the user can post collective operations.

Notes : From the user perspective, the context handle represents a communication resource. The user can create one context and use it for multiple teams or use with a single team. This provides a finer control of resources for the user. From the library implementation perspective, the context could represent the network parallelism. The UCC library implementation can choose to abstract injection queues, network endpoints, GPU device context, UCP worker, or UCP endpoints using the communication context handles. 

