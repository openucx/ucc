/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * Copyright (C) Huawei Technologies Co., Ltd. 2020.  All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TEST_BASE_H
#define UCC_TEST_BASE_H

/* gcc 4.3.4 compilation */
#ifndef UINT8_MAX
#define __STDC_LIMIT_MACROS
#include <stdint.h>
#endif

#define __STDC_FORMAT_MACROS 1
#include <inttypes.h>

#include "test_helpers.h"

#include <core/ucc_global_opts.h>

#include <map>
#include <vector>
#include <string>

namespace ucc {

/* These 10 definitions were grabbed from UCX's log_def.h ... consider importing */
#define UCC_MAX_LOG_HANDLERS (32)

/**
 * Logging levels.
 */
typedef enum {
    UCC_LOG_LEVEL_FATAL,        /* Immediate termination */
    UCC_LOG_LEVEL_ERROR,        /* Error is returned to the user */
    UCC_LOG_LEVEL_WARN,         /* Something's wrong, but we continue */
    UCC_LOG_LEVEL_DIAG,         /* Diagnostics, silent adjustments or internal error handling */
    UCC_LOG_LEVEL_INFO,         /* Information */
    UCC_LOG_LEVEL_DEBUG,        /* Low-volume debugging */
    UCC_LOG_LEVEL_TRACE,        /* High-volume debugging */
    UCC_LOG_LEVEL_TRACE_REQ,    /* Every send/receive request */
    UCC_LOG_LEVEL_TRACE_DATA,   /* Data sent/received on the transport */
    UCC_LOG_LEVEL_TRACE_ASYNC,  /* Asynchronous progress engine */
    UCC_LOG_LEVEL_TRACE_FUNC,   /* Function calls */
    UCC_LOG_LEVEL_TRACE_POLL,   /* Polling functions */
    UCC_LOG_LEVEL_LAST,
    UCC_LOG_LEVEL_PRINT         /* Temporary output */
} ucc_log_level_t;

/**
 * Logging component.
 */
typedef struct ucc_log_component_config {
    ucc_log_level_t log_level;
    char            name[16];
} ucc_log_component_config_t;

typedef enum {
    UCC_LOG_FUNC_RC_STOP,
    UCC_LOG_FUNC_RC_CONTINUE
} ucc_log_func_rc_t;

/**
 * Function type for handling log messages.
 *
 * @param file      Source file name.
 * @param line      Source line number.
 * @param function  Function name.
 * @param level     Log level.
 * @param comp_conf Component specific log config.
 * @param message   Log message - format string.
 * @param ap        Log message format parameters.
 *
 * @return UCC_LOG_FUNC_RC_CONTINUE - continue to next log handler.
 *         UCC_LOG_FUNC_RC_STOP     - don't continue.
 */
typedef ucc_log_func_rc_t (*ucc_log_func_t)(const char *file, unsigned line,
                                            const char *function, ucc_log_level_t level,
                                            const ucc_log_component_config_t *comp_conf,
                                            const char *message, va_list ap);

static unsigned ucc_log_handlers_count = 0;
static ucc_log_func_t ucc_log_handlers[UCC_MAX_LOG_HANDLERS];

static inline void ucc_log_push_handler(ucc_log_func_t handler)
{
    if (ucc_log_handlers_count < UCC_MAX_LOG_HANDLERS) {
        ucc_log_handlers[ucc_log_handlers_count++] = handler;
    }
}

static inline void ucc_log_pop_handler()
{
    if (ucc_log_handlers_count > 0) {
        --ucc_log_handlers_count;
    }
}

static inline unsigned ucc_log_num_handlers()
{
    return ucc_log_handlers_count;
}

/**
 * Base class for tests
 */
class test_base {
public:
    typedef enum {
        IGNORE_IF_NOT_EXIST,
        FAIL_IF_NOT_EXIST,
        SETENV_IF_NOT_EXIST,
        SKIP_IF_NOT_EXIST
    } modify_config_mode_t;

    test_base();
    virtual ~test_base();

protected:
    class scoped_log_handler {
    public:
        scoped_log_handler(ucc_log_func_t handler) {
            ucc_log_push_handler(handler);
        }
        ~scoped_log_handler() {
            ucc_log_pop_handler();
        }
    };

    typedef enum {
        NEW, RUNNING, SKIPPED, ABORTED, FINISHED
    } state_t;

    typedef std::vector<ucc_global_config_t> config_stack_t;

    void SetUpProxy();
    void TearDownProxy();
    void TestBodyProxy();
    static std::string format_message(const char *message, va_list ap);

    virtual void cleanup();
    virtual void init();
    bool barrier();

    virtual void check_skip_test() = 0;

    virtual void test_body() = 0;

    static ucc_log_func_rc_t
    count_warns_logger(const char *file, unsigned line, const char *function,
                       ucc_log_level_t level,
                       const ucc_log_component_config_t *comp_conf,
                       const char *message, va_list ap);

    static ucc_log_func_rc_t
    hide_errors_logger(const char *file, unsigned line, const char *function,
                       ucc_log_level_t level,
                       const ucc_log_component_config_t *comp_conf,
                       const char *message, va_list ap);

    static ucc_log_func_rc_t
    hide_warns_logger(const char *file, unsigned line, const char *function,
                      ucc_log_level_t level,
                      const ucc_log_component_config_t *comp_conf,
                      const char *message, va_list ap);

    static ucc_log_func_rc_t
    wrap_errors_logger(const char *file, unsigned line, const char *function,
                       ucc_log_level_t level,
                       const ucc_log_component_config_t *comp_conf,
                       const char *message, va_list ap);

    unsigned num_errors();

    unsigned num_warnings();

    state_t                         m_state;
    bool                            m_initialized;
    config_stack_t                  m_config_stack;
    int                             m_num_valgrind_errors_before;
    unsigned                        m_num_errors_before;
    unsigned                        m_num_warnings_before;
    unsigned                        m_num_log_handlers_before;

    static pthread_mutex_t          m_logger_mutex;
    static unsigned                 m_total_errors;
    static unsigned                 m_total_warnings;
    static std::vector<std::string> m_errors;
    static std::vector<std::string> m_warnings;
    static std::vector<std::string> m_first_warns_and_errors;

private:
    void skipped(const test_skip_exception& e);
    void run();
    static void push_debug_message_with_limit(std::vector<std::string>& vec,
                                              const std::string& message,
                                              const size_t limit);

    static void *thread_func(void *arg);

    pthread_barrier_t    m_barrier;
};

#define UCC_TEST_BASE_IMPL \
    virtual void SetUp() { \
        test_base::SetUpProxy(); \
    } \
    \
    virtual void TearDown() { \
        test_base::TearDownProxy(); \
    } \
    virtual void TestBody() { \
        test_base::TestBodyProxy(); \
    }

/*
 * Base class from generic tests
 */
class test : public testing::Test, public test_base {
public:
    UCC_TEST_BASE_IMPL;
};

/*
 * Base class from generic tests with user-defined parameter
 */
template <typename T>
class test_with_param : public testing::TestWithParam<T>, public test_base {
public:
    UCC_TEST_BASE_IMPL;
};

}

/*
 * Helper macro
 */
#define UCC_TEST_(test_case_name, test_name, parent_id, \
                  num_threads, skip_cond, skip_reason, ...) \
class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) : public test_case_name { \
 public: \
  GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() { \
  } \
 protected: \
  virtual void init() { \
      test_case_name::init(); \
  } \
 private: \
  virtual void check_skip_test() { \
      if (skip_cond) { \
          UCC_TEST_SKIP_R(skip_reason); \
      } \
  } \
  virtual void test_body(); \
  static ::testing::TestInfo* const test_info_;\
  GTEST_DISALLOW_COPY_AND_ASSIGN_(\
      GTEST_TEST_CLASS_NAME_(test_case_name, test_name));\
}; \
\
::testing::TestInfo* const GTEST_TEST_CLASS_NAME_(test_case_name, test_name)\
  ::test_info_ = \
    ::testing::internal::MakeAndRegisterTestInfo( \
        #test_case_name, \
        (num_threads == 1) ? #test_name : #test_name "/mt_" #num_threads, \
        "", "", \
        (parent_id), \
        test_case_name::SetUpTestCase, \
        test_case_name::TearDownTestCase, \
        new ::testing::internal::TestFactoryImpl< \
            GTEST_TEST_CLASS_NAME_(test_case_name, test_name)>); \
void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::test_body()


/*
 * Define test fixture with modified configuration
 */
#define UCC_TEST_F(test_fixture, test_name, ...) \
  UCC_TEST_(test_fixture, test_name, \
            ::testing::internal::GetTypeId<test_fixture>(), \
            1, 0, "", __VA_ARGS__)


/*
 * Define test fixture with modified configuration and check skip condition
 */
#define UCC_TEST_SKIP_COND_F(test_fixture, test_name, skip_cond, ...) \
  UCC_TEST_(test_fixture, test_name, \
            ::testing::internal::GetTypeId<test_fixture>(), \
            1, skip_cond, #skip_cond, __VA_ARGS__)


/*
 * Define test fixture with multiple threads
 */
#define UCC_MT_TEST_F(test_fixture, test_name, num_threads, ...) \
  UCC_TEST_(test_fixture, test_name, \
            ::testing::internal::GetTypeId<test_fixture>(), \
            num_threads, 0, "", __VA_ARGS__)


/*
 * Helper macro
 */
#define UCC_TEST_P_(test_case_name, test_name, num_threads, \
                    skip_cond, skip_reason, ...) \
  class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) \
      : public test_case_name { \
   public: \
    GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() { \
       set_num_threads(num_threads); \
    } \
    virtual void test_body(); \
   protected: \
    virtual void init() { \
        UCC_PP_FOREACH(UCC_TEST_SET_CONFIG, _, __VA_ARGS__) \
        test_case_name::init(); \
    } \
   private: \
    virtual void check_skip_test() { \
        if (skip_cond) { \
            UCC_TEST_SKIP_R(skip_reason); \
        } \
    } \
    static int AddToRegistry() { \
        ::testing::UnitTest::GetInstance()->parameterized_test_registry(). \
            GetTestCasePatternHolder<test_case_name>( \
                #test_case_name, __FILE__, __LINE__)->AddTestPattern( \
                    #test_case_name, \
                    (num_threads == 1) ? #test_name : #test_name "/mt_" #num_threads, \
                    new ::testing::internal::TestMetaFactory< \
                        GTEST_TEST_CLASS_NAME_(test_case_name, test_name)>()); \
        return 0; \
    } \
    static int gtest_registering_dummy_; \
    GTEST_DISALLOW_COPY_AND_ASSIGN_(\
        GTEST_TEST_CLASS_NAME_(test_case_name, test_name)); \
  }; \
  int GTEST_TEST_CLASS_NAME_(test_case_name, \
                             test_name)::gtest_registering_dummy_ = \
      GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::AddToRegistry(); \
  void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::test_body()


/*
 * Define parameterized test with modified configuration
 */
#define UCC_TEST_P(test_case_name, test_name, ...) \
    UCC_TEST_P_(test_case_name, test_name, 1, 0, "", __VA_ARGS__)


/*
 * Define parameterized test with modified configuration and check skip condition
 */
#define UCC_TEST_SKIP_COND_P(test_case_name, test_name, skip_cond, ...) \
    UCC_TEST_P_(test_case_name, test_name, 1, skip_cond, #skip_cond, __VA_ARGS__)


/*
 * Define parameterized test with multiple threads
 */
#define UCC_MT_TEST_P(test_case_name, test_name, num_threads, ...) \
    UCC_TEST_P_(test_case_name, test_name, num_threads, 0, "", __VA_ARGS__)

#endif
