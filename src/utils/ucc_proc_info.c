#include "ucc_proc_info.h"
ucc_proc_info_t ucc_local_proc;
char ucc_local_hostname[128];

ucc_status_t ucc_local_proc_info_init()
{
    ucc_local_proc.host_id = gethostid();
    if (gethostname(ucc_local_hostname, sizeof(ucc_local_hostname))) {
        ucc_local_hostname[0] = '\0';
    }
    return UCC_OK;
}
