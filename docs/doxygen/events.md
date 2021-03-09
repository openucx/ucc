### Execution Engine and Events

The execution engine is an execution context that supports event-driven network
execution on the CUDA streams, CPU threads, and DPU threads. It is intended to
interact with execution threads that are asynchronous (offloaded collective
execution) which can be implemented on GPUs, DPUs, or remote CPUs.  

UCC supports triggering collective operations by library-generated and
user-generated events. The library events are generated on posting or completion
of operations. The user-generated events include the completion of compute or
communication operations. With a combination of library-generated and
user-generated events, one can build dependencies between compute and collective
operations, or between the collective operations.

Besides the execution engine, events are key for
event-driven execution. The operations on the execution engines generate events
that are stored internally on the execution engines.
The valid events are defined by @ref ucc\_event\_type\_t. If the underlying
hardware doesn't support event-driven execution, the implementations
can implement this with the event queues or lists.

The interaction between the user and library is through the UCC interfaces. @ref
ucc\_ee\_create creates execution engines. The user or
library can generate an event and post it to the execution engines using @ref
ucc\_ee\_set\_event interface. The user can wait on the events with the @ref
ucc\_ee\_wait interface. The user can get the event from the ee using
ucc\_ee\_get\_event interface and acknowledge the event with ucc\_ee\_ack\_event
interface. Once acknowledged, the library destroys the event.

Thread Mode: While in the UCC\_THREAD\_MULTIPLE mode, the execution engine and
operations can be invoked from multiple threads.

Order: All non-triggered operations posted to the execution engine are executed
in-order. However, there are no ordering guarantees between the execution
engines.

### Triggered Operations

Triggered operations enable the posting of operations on an event. For triggered
operations, the team should be configured with event-driven execution. The
collection operations is defined by the interface @ref
ucc\_collective\_triggered\_post.

The operations are launched on the event. So, there is no order established by
the library. If user desires an order for the triggered operations, the
user should provide the tag for matching the collective operations.

### Interaction between an User Thread and Event-driven UCC

The figure shows the interaction between application threads and the UCC library
configured with event-driven teams. In this example scenario, we assume that the
UCC team are configured with two events queues - one for post operations and one
for completions.

(1) The application initializes the collective operation when it knows the
control parameters of the collective such as buffer addresses, lengths, and
participants of the collective. The data need not be ready as it posts the
collective operation which will be triggered on an event. For example, the event
here is the completion of compute by the application.

(2) When the application completes the compute, it posts the
UCC\_EVENT\_COMPUTE\_COMPLETE event to the execution engine.

(3) The library thread polls the event queue and triggers the operations that
are related to the compute event.

(4) The library posts the UCC\_EVENT\_POST\_COMPLETE event to the event queue.

(5) On completion of the collective operation, the library posts
UCC\_EVENT\_COLLECTIVE\_COMPLETE event to the completion event queue.

\image latex ucc\_events.pdf "UCC Execution Engine and Events"
\image html ucc\_events.png "UCC Execution Engine and Events"


