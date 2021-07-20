#include <ucc/api/ucc.h>
#include <cassert>

int main(int argc, char *argv[]) {
  ucc_status_t st;

  ucc_lib_config_h lib_config;
  st = ucc_lib_config_read(nullptr, nullptr, &lib_config);
  assert(st == UCC_OK);

  ucc_lib_params_t lib_params = {};
  lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
  lib_params.thread_mode = UCC_THREAD_MULTIPLE;

  ucc_lib_h lib;
  st = ucc_init(&lib_params, lib_config, &lib);
  assert(st == UCC_OK);

  ucc_lib_config_release(lib_config);
  ucc_finalize(lib);
  return 0;
}
