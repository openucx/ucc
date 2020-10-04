## Library Initialization and Finalization

These routines are responsible for allocating, initializing, and finalizing the resources for the library.

The UCC can be configured in three thread modes UCC\_THREAD\_SINGLE, UCC\_THREAD\_FUNNELED, and UCC\_LIB\_THREAD\_MULTIPLE. In the UCC\_THREAD\_SINGLE mode, the user program must not be multithreaded. In the UCC\_THREAD\_FUNNELED mode, the user program may be multithreaded. However, all UCC interfaces should be invoked from the same thread. In the UCC\_THREAD\_MULTIPLE mode, the user program can be multithreaded and any thread may invoke the UCC operations.

The user can request different types of collective operations that vary in their synchronization models. The valid synchronization models are UCC\_NO\_SYNC\_COLLECTIVES and UCC\_SYNC\_COLLECTIVES. The details of these synchronization models are described in the collective operation section.

The user can request the different collective operations and reduction operations required.  The complete set of valid collective operations and reduction types are defined with the structures ucc\_coll\_type\_t and ucc\_reduction\_op\_t.

