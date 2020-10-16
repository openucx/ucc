/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2012.  ALL RIGHTS RESERVED.
 * Copyright (C) Huawei Technologies Co., Ltd. 2020.  All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "test_helpers.h"

#include <sys/time.h>
#include <set>

extern "C" {
// On some platforms users have to declare environ explicitly
extern char** environ;
}

namespace ucc {

typedef std::pair<std::string, ::testing::TimeInMillis> test_result_t;

const double test_timeout_in_sec = 60.;

const double watchdog_timeout_default = 900.; // 15 minutes

static test_watchdog_t watchdog;

std::set< const ::testing::TestInfo*> skipped_tests;

/* These 2 definitions were grabbed from UCX's preprocessor.h ... consider importing */
#define __UCS_TOKENPASTE_HELPER(x, y)     x ## y
#define UCS_PP_TOKENPASTE(x, y)           __UCS_TOKENPASTE_HELPER(x, y)

/* These 10 definitions were grabbed from UCX's time.h ... consider importing */

#define UCS_MSEC_PER_SEC   1000ull       /* Milli */
#define UCS_USEC_PER_SEC   1000000ul     /* Micro */
#define UCS_NSEC_PER_SEC   1000000000ul  /* Nano */
typedef unsigned long   ucs_time_t;

/**
 * @return The current time, in UCS time units.
 */
static inline ucs_time_t ucs_get_time()
{
    struct timeval tv;

    if (gettimeofday(&tv, NULL) != 0) {
    return 0;
    }
    return ((((uint64_t)(tv.tv_sec)) * 1000000ULL) + ((uint64_t)(tv.tv_usec)));
}

/**
 * @return The current accurate time, in seconds.
 * @note This function may have higher overhead than @ref ucs_get_time()
 */
static inline double ucs_get_accurate_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (tv.tv_usec / (double)UCS_USEC_PER_SEC);
}

/* Convert seconds to POSIX timespec */
static inline void ucs_sec_to_timespec(double seconds, struct timespec *ts)
{
    int64_t nsec = (int64_t)( (seconds * UCS_NSEC_PER_SEC) + 0.5 );
    ts->tv_sec  = nsec / UCS_NSEC_PER_SEC;
    ts->tv_nsec = nsec % UCS_NSEC_PER_SEC;
}

/**
 * @return The clock value of a single second.
 */
static inline double ucs_time_sec_value()
{
    return 1.0E6;
}

/**
 * Convert seconds to UCS time units.
 */
static inline ucs_time_t ucs_time_from_sec(double sec)
{
    return (ucs_time_t)(sec * ucs_time_sec_value() + 0.5);
}

/**
 * Convert UCS time units to seconds.
 */
static inline double ucs_time_to_sec(ucs_time_t t)
{
    return t / ucs_time_sec_value();
}

/**
 * Convert UCS time units to microseconds.
 */
static inline double ucs_time_to_usec(ucs_time_t t)
{
    return ucs_time_to_sec(t) * UCS_USEC_PER_SEC;
}

void *watchdog_func(void *arg)
{
    int ret = 0;
    double now;
    struct timespec timeout;

    pthread_mutex_lock(&watchdog.mutex);

    // sync with the watched thread
    pthread_barrier_wait(&watchdog.barrier);

    do {
        now = ucs_get_accurate_time();
        ucs_sec_to_timespec(now + watchdog.timeout, &timeout);

        ret = pthread_cond_timedwait(&watchdog.cv, &watchdog.mutex, &timeout);
        if (!ret) {
            pthread_barrier_wait(&watchdog.barrier);
        } else {
            // something wrong happened - handle it
            ADD_FAILURE() << strerror(ret) << " - abort testing";
            if (ret == ETIMEDOUT) {
                pthread_kill(watchdog.watched_thread, watchdog.kill_signal);
            } else {
                abort();
            }
        }

        switch (watchdog.state) {
        case WATCHDOG_TEST:
            watchdog.kill_signal = SIGTERM;
            // reset when the test completed
            watchdog.state = WATCHDOG_DEFAULT_SET;
            break;
        case WATCHDOG_RUN:
            // yawn - nothing to do
            break;
        case WATCHDOG_STOP:
            // force the end of the loop
            ret = 1;
            break;
        case WATCHDOG_TIMEOUT_SET:
            // reset when the test completed
            watchdog.state = WATCHDOG_DEFAULT_SET;
            break;
        case WATCHDOG_DEFAULT_SET:
            watchdog.timeout     = watchdog_timeout_default;
            watchdog.state       = WATCHDOG_RUN;
            watchdog.kill_signal = SIGABRT;
            break;
        }
    } while (!ret);

    pthread_mutex_unlock(&watchdog.mutex);

    return NULL;
}

void watchdog_signal(bool barrier)
{
    pthread_mutex_lock(&watchdog.mutex);
    pthread_cond_signal(&watchdog.cv);
    pthread_mutex_unlock(&watchdog.mutex);

    if (barrier) {
        pthread_barrier_wait(&watchdog.barrier);
    }
}

void watchdog_set(test_watchdog_state_t new_state, double new_timeout)
{
    pthread_mutex_lock(&watchdog.mutex);
    // change timeout value
    watchdog.timeout = new_timeout;
    watchdog.state   = new_state;
    // apply new value for timeout
    watchdog_signal(0);
    pthread_mutex_unlock(&watchdog.mutex);

    pthread_barrier_wait(&watchdog.barrier);
}

void watchdog_set(test_watchdog_state_t new_state)
{
    watchdog_set(new_state, watchdog_timeout_default);
}

void watchdog_set(double new_timeout)
{
    watchdog_set(WATCHDOG_TIMEOUT_SET, new_timeout);
}

#define WATCHDOG_DEFINE_GETTER(_what, _what_type) \
    _what_type UCS_PP_TOKENPASTE(watchdog_get_, _what)() \
    { \
        _what_type value; \
        \
        pthread_mutex_lock(&watchdog.mutex); \
        value = watchdog._what; \
        pthread_mutex_unlock(&watchdog.mutex); \
        \
        return value; \
    }

WATCHDOG_DEFINE_GETTER(timeout, double)
WATCHDOG_DEFINE_GETTER(state, test_watchdog_state_t)
WATCHDOG_DEFINE_GETTER(kill_signal, int)

int watchdog_start()
{
    pthread_mutexattr_t mutex_attr;
    int ret;

    ret = pthread_mutexattr_init(&mutex_attr);
    if (ret != 0) {
        return -1;
    }
    // create reentrant mutex
    ret = pthread_mutexattr_settype(&mutex_attr, PTHREAD_MUTEX_RECURSIVE);
    if (ret != 0) {
        goto err_destroy_mutex_attr;
    }

    ret = pthread_mutex_init(&watchdog.mutex, &mutex_attr);
    if (ret != 0) {
        goto err_destroy_mutex_attr;
    }

    ret = pthread_cond_init(&watchdog.cv, NULL);
    if (ret != 0) {
        goto err_destroy_mutex;
    }

    // 2 - watched thread + watchdog
    ret = pthread_barrier_init(&watchdog.barrier, NULL, 2);
    if (ret != 0) {
        goto err_destroy_cond;
    }

    pthread_mutex_lock(&watchdog.mutex);
    watchdog.state          = WATCHDOG_RUN;
    watchdog.timeout        = watchdog_timeout_default;
    watchdog.kill_signal    = SIGABRT;
    watchdog.watched_thread = pthread_self();
    pthread_mutex_unlock(&watchdog.mutex);

    ret = pthread_create(&watchdog.thread, NULL, watchdog_func, NULL);
    if (ret != 0) {
        goto err_destroy_barrier;
    }

    pthread_mutexattr_destroy(&mutex_attr);

    // sync with the watchdog thread
    pthread_barrier_wait(&watchdog.barrier);

    // test signaling
    watchdog_signal();

    return 0;

err_destroy_barrier:
    pthread_barrier_destroy(&watchdog.barrier);
err_destroy_cond:
    pthread_cond_destroy(&watchdog.cv);
err_destroy_mutex:
    pthread_mutex_destroy(&watchdog.mutex);
err_destroy_mutex_attr:
    pthread_mutexattr_destroy(&mutex_attr);
    return -1;
}

void watchdog_stop()
{
    void *ret_val;

    pthread_mutex_lock(&watchdog.mutex);
    watchdog.state = WATCHDOG_STOP;
    watchdog_signal(0);
    pthread_mutex_unlock(&watchdog.mutex);

    pthread_barrier_wait(&watchdog.barrier);
    pthread_join(watchdog.thread, &ret_val);

    pthread_barrier_destroy(&watchdog.barrier);
    pthread_cond_destroy(&watchdog.cv);
    pthread_mutex_destroy(&watchdog.mutex);
}

static bool test_results_cmp(const test_result_t &a, const test_result_t &b)
{
    return a.second > b.second;
}

void analyze_test_results()
{
    // GTEST_REPORT_LONGEST_TESTS=100 will report TOP-100 longest tests
    /* coverity[tainted_data_return] */
    char *env_p = getenv("GTEST_REPORT_LONGEST_TESTS");
    if (env_p == NULL) {
        return;
    }

    size_t total_skipped_cnt                   = skipped_tests.size();
    ::testing::TimeInMillis total_skipped_time = 0;
    size_t max_name_size                       = 0;
    std::set< const ::testing::TestInfo*>::iterator skipped_it;
    int top_n;

    if (!strcmp(env_p, "*")) {
        top_n = std::numeric_limits<int>::max();
    } else {
        top_n = atoi(env_p);
        if (!top_n) {
            return;
        }
    }

    ::testing::UnitTest *unit_test = ::testing::UnitTest::GetInstance();
    std::vector<test_result_t> test_results;

    if (unit_test == NULL) {
        ADD_FAILURE() << "Unable to get the Unit Test instance";
        return;
    }

    for (int i = 0; i < unit_test->total_test_case_count(); i++) {
        const ::testing::TestCase *test_case = unit_test->GetTestCase(i);
        if (test_case == NULL) {
            ADD_FAILURE() << "Unable to get the Test Case instance with index "
                          << i;
            return;
        }

        for (int i = 0; i < test_case->total_test_count(); i++) {
            const ::testing::TestInfo *test = test_case->GetTestInfo(i);
            if (test == NULL) {
                ADD_FAILURE() << "Unable to get the Test Info instance with index "
                              << i;
                return;
            }

            if (test->should_run()) {
                const ::testing::TestResult *result = test->result();
                std::string test_name               = test->test_case_name();

                test_name += ".";
                test_name += test->name();

                test_results.push_back(std::make_pair(test_name,
                                                      result->elapsed_time()));

                max_name_size = std::max(test_name.size(), max_name_size);

                skipped_it = skipped_tests.find(test);
                if (skipped_it != skipped_tests.end()) {
                    total_skipped_time += result->elapsed_time();
                    skipped_tests.erase(skipped_it);
                }
            }
        }
    }

    std::sort(test_results.begin(), test_results.end(), test_results_cmp);

    top_n = std::min((int)test_results.size(), top_n);
    if (!top_n) {
        return;
    }

    // Print TOP-<N> slowest tests
    int max_index_size = ucc::to_string(top_n).size();
    std::cout << std::endl << "TOP-" << top_n << " longest tests:" << std::endl;

    for (int i = 0; i < top_n; i++) {
        std::cout << std::setw(max_index_size - ucc::to_string(i + 1).size() + 1)
                  << (i + 1) << ". " << test_results[i].first
                  << std::setw(max_name_size - test_results[i].first.size() + 3)
                  << " - " << test_results[i].second << " ms" << std::endl;
    }

    // Print skipped tests statistics
    std::cout << std::endl << "Skipped tests: count - "
              << total_skipped_cnt << ", time - "
              << total_skipped_time << " ms" << std::endl;
}

int test_time_multiplier()
{
    int factor = 1;
#if _BullseyeCoverage
    factor *= 10;
#endif
    /*
    if (RUNNING_ON_VALGRIND) {
        factor *= 20;
    }
    */
    return factor;
}

void fill_random(void *data, size_t size)
{
    if (ucc::test_time_multiplier() > 1) {
        memset(data, 0, size);
        return;
    }

    uint64_t seed = rand();
    for (size_t i = 0; i < size / sizeof(uint64_t); ++i) {
        ((uint64_t*)data)[i] = seed;
        seed = seed * 10 + 17;
    }
    size_t remainder = size % sizeof(uint64_t);
    memset((char*)data + size - remainder, 0xab, remainder);
}

scoped_setenv::scoped_setenv(const char *name, const char *value) : m_name(name) {
    if (getenv(name)) {
        m_old_value = getenv(name);
    }
    setenv(m_name.c_str(), value, 1);
}

scoped_setenv::~scoped_setenv() {
    if (!m_old_value.empty()) {
        setenv(m_name.c_str(), m_old_value.c_str(), 1);
    } else {
        unsetenv(m_name.c_str());
    }
}

#define UCC_DEFAULT_ENV_PREFIX "UCC_"
ucc_env_cleanup::ucc_env_cleanup() {
    const size_t prefix_len = strlen(UCC_DEFAULT_ENV_PREFIX);
    char **envp;

    for (envp = environ; *envp != NULL; ++envp) {
        std::string env_var = *envp;

        if ((env_var.find("=") != std::string::npos) &&
            (env_var.find(UCC_DEFAULT_ENV_PREFIX, 0, prefix_len) != std::string::npos)) {
            ucc_env_storage.push_back(env_var);
        }
    }

    for (size_t i = 0; i < ucc_env_storage.size(); i++) {
        std::string var_name =
            ucc_env_storage[i].substr(0, ucc_env_storage[i].find("="));

        unsetenv(var_name.c_str());
    }
}

ucc_env_cleanup::~ucc_env_cleanup() {
    while (!ucc_env_storage.empty()) {
        std::string var_name =
            ucc_env_storage.back().substr(0, ucc_env_storage.back().find("="));
        std::string var_value =
            ucc_env_storage.back().substr(ucc_env_storage.back().find("=") + 1);

        setenv(var_name.c_str(), var_value.c_str(), 1);
        ucc_env_storage.pop_back();
    }
}

void safe_sleep(double sec) {
    ucs_time_t current_time = ucs_get_time();
    ucs_time_t end_time = current_time + ucs_time_from_sec(sec);

    while (current_time < end_time) {
        usleep((long)ucs_time_to_usec(end_time - current_time));
        current_time = ucs_get_time();
    }
}

void safe_usleep(double usec) {
    safe_sleep(usec * 1e-6);
}

void *mmap_fixed_address() {
    return (void*)0xff0000000;
}

std::string compact_string(const std::string &str, size_t length)
{
    if (str.length() <= length * 2) {
        return str;
    }

    return str.substr(0, length) + "..." + str.substr(str.length() - length);
}

std::string exit_status_info(int exit_status)
{
    std::stringstream ss;

    if (WIFEXITED(exit_status)) {
        ss << ", exited with status " << WEXITSTATUS(exit_status);
    }
    if (WIFSIGNALED(exit_status)) {
        ss << ", signaled with status " << WTERMSIG(exit_status);
    }
    if (WIFSTOPPED(exit_status)) {
        ss << ", stopped with status " << WSTOPSIG(exit_status);
    }

    return ss.str().substr(2, std::string::npos);
}

auto_buffer::auto_buffer(size_t size) : m_ptr(malloc(size)) {
    if (!m_ptr) {
        UCS_TEST_ABORT("Failed to allocate memory");
    }
}

auto_buffer::~auto_buffer()
{
    free(m_ptr);
}

void* auto_buffer::operator*() const {
    return m_ptr;
};

namespace detail {

#define ucs_max(_a, _b) \
({ \
    typeof(_a) _max_a = (_a); \
    typeof(_b) _max_b = (_b); \
    (_max_a > _max_b) ? _max_a : _max_b; \
})

message_stream::message_stream(const std::string& title) {
    static const char PADDING[] = "          ";
    static const size_t WIDTH = strlen(PADDING);

    msg <<  "[";
    msg.write(PADDING, ucs_max(WIDTH - 1, title.length()) - title.length());
    msg << title << " ] ";
}

message_stream::~message_stream() {
    msg << std::endl;
    std::cout << msg.str() << std::flush;
}

} // detail

void skip_on_address_sanitizer()
{
#ifdef __SANITIZE_ADDRESS__
    UCS_TEST_SKIP_R("Address sanitizer");
#endif
}

} // ucs
