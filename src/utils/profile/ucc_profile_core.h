#ifdef HAVE_PROFILING_CORE
#include "utils/profile/ucc_profile.h"
#else
#include "utils/profile/ucc_profile_off.h"
#endif
#define UCC_CORE_PROFILE_FUNC UCC_PROFILE_FUNC
