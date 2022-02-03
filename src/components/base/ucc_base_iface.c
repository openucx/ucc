#include "ucc_base_iface.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_log.h"

ucc_config_field_t ucc_base_lib_config_table[] = {
    {"LOG_LEVEL", "warn",
     "UCC logging level. Messages with a level higher or equal to the "
     "selected will be printed.\n"
     "Possible values are: fatal, error, warn, info, debug, trace, data, func, "
     "poll.",
     ucc_offsetof(ucc_base_lib_config_t, log_component), UCC_CONFIG_TYPE_LOG_COMP},

    {NULL}};

ucc_config_field_t ucc_base_ctx_config_table[] = {
    {"TUNE", "", "Collective tuning modifier for a CL/TL component\n"
     "format: token1#token2#...#tokenn - '#' separated list of tokens where\n"
     "    token=coll_type:msg_range:mem_type:team_size:score:alg - ':' separated\n"
     "    list of qualifiers. Each qualifier is optional. The only requirement\n"
     "    is that either \"score\" or \"alg\" is provided.\n"
     "qualifiers:\n"
     "    coll_type=coll_type_1,coll_type_2,...,coll_type_n - ',' separated\n"
     "              list of coll_types\n"
     "    msg_range=m_start_1-m_end_1,m_start_2-m_end_2,..,m_start_n-m_end_n -\n"
     "              ',' separated list of msg ranges, where each range is\n"
     "              represented by \"start\" and \"end\" values separated by \"-\".\n"
     "              Special value \"inf\" means MAX msg size.\n"
     "    mem_type=m1,m2,..,mn - ',' separated list of memory types\n"
     "    team_size=[t_start_1-t_end_1,t_start_2-t_end_2,...,t_start_n-t_end_n] -\n"
     "              ',' separated list of team size ranges enclosed with [].\n"
     "    score=value - int value from 0 to \"inf\"\n"
     "          0 - disables the CL/TL in the given range for a given coll\n"
     "          inf - forces the CL/TL in the given range for a given coll\n"
     "    alg=@<value|str> - character @ followed by either int number or string\n"
     "        representing the collective algorithm.",
     ucc_offsetof(ucc_base_ctx_config_t, score_str), UCC_CONFIG_TYPE_STRING},

    {NULL}};

ucc_status_t ucc_base_config_read(const char *full_prefix,
                                  ucc_config_global_list_entry_t *cfg_entry,
                                  ucc_base_config_t **config)
{
    ucc_base_config_t *cfg;
    ucc_status_t       status;
    cfg = ucc_malloc(cfg_entry->size, "cl_ctx_cfg");
    if (!cfg) {
        ucc_error("failed to allocate %zd bytes for cl context config",
                  cfg_entry->size);
        return UCC_ERR_NO_MEMORY;
    }
    cfg->cfg_entry = cfg_entry;
    status = ucc_config_parser_fill_opts(cfg, cfg_entry->table, full_prefix,
                                         cfg_entry->prefix, 0);
    if (UCC_OK != status) {
        ucc_free(cfg);
        *config = NULL;
    } else {
        *config = cfg;
    }
    return status;
}
