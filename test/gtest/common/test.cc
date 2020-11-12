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

unsigned test_base::m_total_errors = 0;
std::vector<std::string> test_base::m_errors;

test_base::test_base() :
                m_state(NEW),
                m_initialized(false),
                m_num_valgrind_errors_before(0),
                m_num_errors_before(0),
                m_num_log_handlers_before(0) {
}

test_base::~test_base() {
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

unsigned test_base::num_errors()
{
    return m_total_errors - m_num_errors_before;
}

void test_base::SetUpProxy() {
    m_num_errors_before          = m_total_errors;

    m_errors.clear();

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

    if (num_errors() > 0) {
        ADD_FAILURE() << "Got " << num_errors() << " errors during the test";
        for (size_t i = 0; i < m_errors.size(); ++i) {
            UCC_TEST_MESSAGE << "< " << m_errors[i] << " >";
        }
    }

    m_errors.clear();
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
