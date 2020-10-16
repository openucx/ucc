/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
 * Copyright (C) Huawei Technologies Co., Ltd. 2020.  All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "test.h"

#include <memory>

namespace ucc {

pthread_mutex_t test_base::m_logger_mutex = PTHREAD_MUTEX_INITIALIZER;
unsigned test_base::m_total_warnings = 0;
unsigned test_base::m_total_errors   = 0;
std::vector<std::string> test_base::m_errors;
std::vector<std::string> test_base::m_warnings;
std::vector<std::string> test_base::m_first_warns_and_errors;

test_base::test_base() :
                m_state(NEW),
                m_initialized(false),
                m_num_threads(1),
                m_num_valgrind_errors_before(0),
                m_num_errors_before(0),
                m_num_warnings_before(0),
                m_num_log_handlers_before(0)
{
    //push_config();
}

test_base::~test_base() {
    //while (!m_config_stack.empty()) {
    //    pop_config();
    //}
}

void test_base::set_num_threads(unsigned num_threads) {
    if (m_state != NEW) {
        GTEST_FAIL() << "Cannot modify number of threads after test is started, "
                     << "it must be done in the constructor.";
    }
    m_num_threads = num_threads;
}

unsigned test_base::num_threads() const {
    return m_num_threads;
}

//void test_base::set_config(const std::string& config_str)
//{
//    std::string::size_type pos = config_str.find("=");
//    modify_config_mode_t mode;
//    std::string name, value;
//
//    if (pos == std::string::npos) {
//        name  = config_str;
//        value = "";
//    } else {
//        name  = config_str.substr(0, pos);
//        value = config_str.substr(pos + 1);
//    }
//
//    mode = FAIL_IF_NOT_EXIST;
//
//    /*
//     * What happens if NAME is not a valid configuration key?
//     * - "NAME=VALUE"   : fail the test
//     * - "NAME?=VALUE"  : set UCX_NAME environment variable
//     * - "NAME~=VALUE"  : skip the test
//     */
//    if (name.length() > 0) {
//        char modifier = name.at(name.length() - 1);
//        bool valid_modifier;
//        switch (modifier) {
//        case '?':
//            mode           = SETENV_IF_NOT_EXIST;
//            valid_modifier = true;
//            break;
//        case '~':
//            mode           = SKIP_IF_NOT_EXIST;
//            valid_modifier = true;
//            break;
//        default:
//            valid_modifier = false;
//            break;
//        }
//        if (valid_modifier) {
//            name = name.substr(0, name.length() - 1);
//        }
//    }
//
//    modify_config(name, value, mode);
//}
//
//void test_base::get_config(const std::string& name, std::string& value, size_t max)
//{
//    ucs_status_t status;
//
//    value.resize(max, '\0');
//    status = ucs_global_opts_get_value(name.c_str(),
//                                       const_cast<char*>(value.c_str()),
//                                       max);
//    if (status != UCS_OK) {
//        GTEST_FAIL() << "Invalid UCS configuration for " << name
//                     << ": " << ucs_status_string(status)
//                     << "(" << status << ")";
//    }
//}
//
//void test_base::modify_config(const std::string& name, const std::string& value,
//                              modify_config_mode_t mode)
//{
//    ucs_status_t status = ucs_global_opts_set_value(name.c_str(), value.c_str());
//    if (status == UCS_ERR_NO_ELEM) {
//        switch (mode) {
//        case FAIL_IF_NOT_EXIST:
//            GTEST_FAIL() << "Invalid UCS configuration for " << name << " : "
//                         << value << ", error message: "
//                         << ucs_status_string(status) << "(" << status << ")";
//        case SETENV_IF_NOT_EXIST:
//            m_env_stack.push_back(new scoped_setenv(("UCX_" + name).c_str(),
//                                                    value.c_str()));
//            break;
//        case SKIP_IF_NOT_EXIST:
//            UCS_TEST_SKIP_R(name + " is not a valid configuration");
//        case IGNORE_IF_NOT_EXIST:
//            break;
//        }
//    } else {
//        ASSERT_UCS_OK(status);
//    }
//}
//
//void test_base::push_config()
//{
//    ucc_global_config_t new_opts;
//    /* save current options to the vector
//     * it is important to keep the first original global options at the first
//     * vector element to release it at the end. Otherwise, memtrack will not work
//     */
//    m_config_stack.push_back(ucc_global_config);
//    ucc_global_config_clone(&new_opts);
//    ucc_global_config = new_config;
//}
//
//void test_base::pop_config()
//{
//    ucs_global_opts_release();
//    ucs_global_opts = m_config_stack.back();
//    m_config_stack.pop_back();
//}

ucs_log_func_rc_t
test_base::count_warns_logger(const char *file, unsigned line, const char *function,
                              ucs_log_level_t level,
                              const ucs_log_component_config_t *comp_conf,
                              const char *message, va_list ap)
{
    pthread_mutex_lock(&m_logger_mutex);
    if (level == UCS_LOG_LEVEL_ERROR) {
        ++m_total_errors;
    } else if (level == UCS_LOG_LEVEL_WARN) {
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
    return UCS_LOG_FUNC_RC_CONTINUE;
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
        UCS_TEST_ABORT("aborting after " + ucc::to_string(vec.size()) +
                       " error messages (" + message + ")");
    }

    vec.push_back(message);
}

ucs_log_func_rc_t
test_base::hide_errors_logger(const char *file, unsigned line, const char *function,
                              ucc::ucs_log_level_t level,
                              const ucs_log_component_config_t *comp_conf,
                              const char *message, va_list ap)
{
    if (level == UCS_LOG_LEVEL_ERROR) {
        pthread_mutex_lock(&m_logger_mutex);
        va_list ap2;
        va_copy(ap2, ap);
        m_errors.push_back(format_message(message, ap2));
        va_end(ap2);
        level = UCS_LOG_LEVEL_DEBUG;
        pthread_mutex_unlock(&m_logger_mutex);
    }

    /* TODO: uncomment when the logging in UCC is ready...
    ucs_log_default_handler(file, line, function, level,
                            &ucc_global_config.log_component, message, ap); */
    return UCS_LOG_FUNC_RC_STOP;
}

ucs_log_func_rc_t
test_base::hide_warns_logger(const char *file, unsigned line, const char *function,
                             ucc::ucs_log_level_t level,
                             const ucs_log_component_config_t *comp_conf,
                             const char *message, va_list ap)
{
    if (level == UCS_LOG_LEVEL_WARN) {
        pthread_mutex_lock(&m_logger_mutex);
        va_list ap2;
        va_copy(ap2, ap);
        m_warnings.push_back(format_message(message, ap2));
        va_end(ap2);
        level = UCS_LOG_LEVEL_DEBUG;
        pthread_mutex_unlock(&m_logger_mutex);
    }

    /* TODO: uncomment when the logging in UCC is ready...
    ucs_log_default_handler(file, line, function, level,
                            &ucc_global_config.log_component, message, ap); */
    return UCS_LOG_FUNC_RC_STOP;
}

ucs_log_func_rc_t
test_base::wrap_errors_logger(const char *file, unsigned line, const char *function,
                              ucs_log_level_t level,
                              const ucs_log_component_config_t *comp_conf,
                              const char *message, va_list ap)
{
    /* Ignore warnings about empty memory pool */
    if (level == UCS_LOG_LEVEL_ERROR) {
        pthread_mutex_lock(&m_logger_mutex);
        std::istringstream iss(format_message(message, ap));
        std::string text;
        while (getline(iss, text, '\n')) {
            push_debug_message_with_limit(m_errors, text, 1000);
            UCS_TEST_MESSAGE << "< " << text << " >";
        }
        pthread_mutex_unlock(&m_logger_mutex);
        return UCS_LOG_FUNC_RC_STOP;
    }

    return UCS_LOG_FUNC_RC_CONTINUE;
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
    m_num_log_handlers_before    = ucs_log_num_handlers();
    ucs_log_push_handler(count_warns_logger);

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
    watchdog_signal();

    if (m_initialized) {
        cleanup();
    }

    m_errors.clear();

    ucs_log_pop_handler();

    unsigned num_not_removed = ucs_log_num_handlers() - m_num_log_handlers_before;
    if (num_not_removed != 0) {
         ADD_FAILURE() << num_not_removed << " log handlers were not removed";
    }

    if ((num_errors() > 0) || (num_warnings() > 0)) {
        ADD_FAILURE() << "Got " << num_errors() << " errors "
                      << "and " << num_warnings() << " warnings "
                      << "during the test";
        for (size_t i = 0; i < m_first_warns_and_errors.size(); ++i) {
            UCS_TEST_MESSAGE << "< " << m_first_warns_and_errors[i] << " >";
        }
    }
}

void test_base::run()
{
    if (num_threads() == 1) {
        test_body();
    } else {
        pthread_t threads[num_threads()];
        pthread_barrier_init(&m_barrier, NULL, num_threads());
        for (unsigned i = 0; i < num_threads(); ++i) {
            pthread_create(&threads[i], NULL, thread_func, reinterpret_cast<void*>(this));
        }
        for (unsigned i = 0; i < num_threads(); ++i) {
            void *retval;
            pthread_join(threads[i], &retval);
        }
        pthread_barrier_destroy(&m_barrier);
    }
}

void *test_base::thread_func(void *arg)
{
    test_base *self = reinterpret_cast<test_base*>(arg);
    self->barrier(); /* Let all threads start in the same time */
    self->test_body();
    return NULL;
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

bool test_base::barrier() {
    int ret = pthread_barrier_wait(&m_barrier);
    if (ret == 0) {
        return false;
    } else if (ret == PTHREAD_BARRIER_SERIAL_THREAD) {
        return true;
    } else {
        UCS_TEST_ABORT("pthread_barrier_wait() failed");
    }
}

}
