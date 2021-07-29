#include "ucc_proc_info.h"
#include "ucc_math.h"
#include "ucc_log.h"
#include <limits.h>
#include <stdint.h>

ucc_proc_info_t ucc_local_proc;
static char ucc_local_hostname[HOST_NAME_MAX];

const char*  ucc_hostname()
{
    return ucc_local_hostname;
}

ucc_status_t ucc_local_proc_info_init()
{
    ucc_local_proc.host_hash = gethostid();
    if (gethostname(ucc_local_hostname, sizeof(ucc_local_hostname))) {
        ucc_warn("couldn't get local hostname");
        ucc_local_hostname[0] = '\0';
    } else {
        strtok(ucc_local_hostname, ".");
        ucc_assert(sizeof(ucc_host_id_t) >= sizeof(unsigned long));
        //TODO: switch to ucs_machine_guid when it is available
        ucc_local_proc.host_hash = ucc_str_hash_djb2(ucc_local_hostname);
    }
    ucc_local_proc.pid       = getpid();
    ucc_local_proc.socket_id = -1;

    ucc_debug("proc pid %d, host %s, host_hash %lu",
              ucc_local_proc.pid, ucc_local_hostname, ucc_local_proc.host_hash);
    return UCC_OK;
}
