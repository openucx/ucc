/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
 * Copyright (C) Huawei Technologies Co., Ltd. 2020.  All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "test.h"

#include <memory>

namespace ucc {

std::set< const ::testing::TestInfo*> skipped_tests;

pthread_mutex_t test_base::m_logger_mutex = PTHREAD_MUTEX_INITIALIZER;
unsigned test_base::m_total_warnings = 0;
unsigned test_base::m_total_errors   = 0;
std::vector<std::string> test_base::m_errors;
std::vector<std::string> test_base::m_warnings;
std::vector<std::string> test_base::m_first_warns_and_errors;

test_base::test_base() :
                m_state(NEW),
                m_initialized(false),
                m_num_valgrind_errors_before(0),
                m_num_errors_before(0),
                m_num_warnings_before(0),
                m_num_log_handlers_before(0) {
}

test_base::~test_base() {
}

ucc_log_func_rc_t
test_base::count_warns_logger(const char *file, unsigned line, const char *function,
                              ucc_log_level_t level,
                              const ucc_log_component_config_t *comp_conf,
                              const char *message, va_list ap)
{
    pthread_mutex_lock(&m_logger_mutex);
    if (level == UCC_LOG_LEVEL_ERROR) {
        ++m_total_errors;
    } else if (level == UCC_LOG_LEVEL_WARN) {
        ++m_total_warnings;
    }
    if (m_first_warns_and_errors.size() < 5) {
        /* Save the first few errors/warnings which cause the test to fail */
        va_list ap2;
        va_copy(ap2, ap);
        std::stringstream ss;
        ss << file << ":" << line << " " << format_message(message, ap2);
        va_end(ap2);
        m_first_warns_and_errors.push_back(ss.str());
    }
    pthread_mutex_unlock(&m_logger_mutex);
    return UCC_LOG_FUNC_RC_CONTINUE;
}

std::string test_base::format_message(const char *message, va_list ap)
{
    const size_t buffer_size = ucs_log_get_buffer_size();
    std::string buf(buffer_size, '\0');
    vsnprintf(&buf[0], buffer_size, message, ap);
    buf.resize(strlen(buf.c_str()));
    return buf;
}

void test_base::push_debug_message_with_limit(std::vector<std::string>& vec,
                                              const std::string& message,
                                              const size_t limit) {
    if (vec.size() >= limit) {
        UCC_TEST_ABORT("aborting after " + ucc::to_string(vec.size()) +
                       " error messages (" + message + ")");
    }

    vec.push_back(message);
}

ucc_log_func_rc_t
test_base::hide_errors_logger(const char *file, unsigned line, const char *function,
                              ucc::ucc_log_level_t level,
                              const ucc_log_component_config_t *comp_conf,
                              const char *message, va_list ap)
{
    if (level == UCC_LOG_LEVEL_ERROR) {
        pthread_mutex_lock(&m_logger_mutex);
        va_list ap2;
        va_copy(ap2, ap);
        m_errors.push_back(format_message(message, ap2));
        va_end(ap2);
        level = UCC_LOG_LEVEL_DEBUG;
        pthread_mutex_unlock(&m_logger_mutex);
    }

    /* TODO: uncomment when the logging in UCC is ready...
    ucc_log_default_handler(file, line, function, level,
                            &ucc_global_config.log_component, message, ap); */
    return UCC_LOG_FUNC_RC_STOP;
}

ucc_log_func_rc_t
test_base::hide_warns_logger(const char *file, unsigned line, const char *function,
                             ucc::ucc_log_level_t level,
                             const ucc_log_component_config_t *comp_conf,
                             const char *message, va_list ap)
{
    if (level == UCC_LOG_LEVEL_WARN) {
        pthread_mutex_lock(&m_logger_mutex);
        va_list ap2;
        va_copy(ap2, ap);
        m_warnings.push_back(format_message(message, ap2));
        va_end(ap2);
        level = UCC_LOG_LEVEL_DEBUG;
        pthread_mutex_unlock(&m_logger_mutex);
    }

    /* TODO: uncomment when the logging in UCC is ready...
    ucc_log_default_handler(file, line, function, level,
                            &ucc_global_config.log_component, message, ap); */
    return UCC_LOG_FUNC_RC_STOP;
}

ucc_log_func_rc_t
test_base::wrap_errors_logger(const char *file, unsigned line, const char *function,
                              ucc_log_level_t level,
                              const ucc_log_component_config_t *comp_conf,
                              const char *message, va_list ap)
{
    /* Ignore warnings about empty memory pool */
    if (level == UCC_LOG_LEVEL_ERROR) {
        pthread_mutex_lock(&m_logger_mutex);
        std::istringstream iss(format_message(message, ap));
        std::string text;
        while (getline(iss, text, '\n')) {
            push_debug_message_with_limit(m_errors, text, 1000);
            UCC_TEST_MESSAGE << "< " << text << " >";
        }
        pthread_mutex_unlock(&m_logger_mutex);
        return UCC_LOG_FUNC_RC_STOP;
    }

    return UCC_LOG_FUNC_RC_CONTINUE;
}

unsigned test_base::num_errors()
{
    return m_total_errors - m_num_errors_before;
}

unsigned test_base::num_warnings()
{
    return m_total_warnings - m_num_warnings_before;
}

void test_base::SetUpProxy() {
    m_num_warnings_before        = m_total_warnings;
    m_num_errors_before          = m_total_errors;

    m_errors.clear();
    m_warnings.clear();
    m_first_warns_and_errors.clear();
    m_num_log_handlers_before    = ucc_log_num_handlers();
    ucc_log_push_handler(count_warns_logger);

    try {
        check_skip_test();
        init();
        m_initialized = true;
        m_state = RUNNING;
    } catch (test_skip_exception& e) {
        skipped(e);
    } catch (test_abort_exception&) {
        m_state = ABORTED;
    }
}

void test_base::TearDownProxy() {
    if (m_initialized) {
        cleanup();
    }

    m_errors.clear();

    ucc_log_pop_handler();

    unsigned num_not_removed = ucc_log_num_handlers() - m_num_log_handlers_before;
    if (num_not_removed != 0) {
         ADD_FAILURE() << num_not_removed << " log handlers were not removed";
    }

    if ((num_errors() > 0) || (num_warnings() > 0)) {
        ADD_FAILURE() << "Got " << num_errors() << " errors "
                      << "and " << num_warnings() << " warnings "
                      << "during the test";
        for (size_t i = 0; i < m_first_warns_and_errors.size(); ++i) {
            UCC_TEST_MESSAGE << "< " << m_first_warns_and_errors[i] << " >";
        }
    }
}

void test_base::run()
{
    test_body();
}

void test_base::TestBodyProxy() {
    if (m_state == RUNNING) {
        try {
            run();
            m_state = FINISHED;
        } catch (test_skip_exception& e) {
            skipped(e);
        } catch (test_abort_exception&) {
            m_state = ABORTED;
        } catch (exit_exception& e) {
            /* If not running on valgrind / execp failed, use exit() */
            exit(e.failed() ? 1 : 0);
        } catch (...) {
            m_state = ABORTED;
            throw;
        }
    }
}

void test_base::skipped(const test_skip_exception& e) {
    std::string reason = e.what();
    if (reason.empty()) {
        detail::message_stream("SKIP");
    } else {
        detail::message_stream("SKIP") << "(" << reason << ")";
    }
    m_state = SKIPPED;
    skipped_tests.insert(::testing::UnitTest::
                         GetInstance()->current_test_info());
}

void test_base::init() {
}

void test_base::cleanup() {
}

} // ucc
