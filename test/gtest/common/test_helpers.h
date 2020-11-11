/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
 * Copyright (C) Huawei Technologies Co., Ltd. 2020.  All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TEST_HELPERS_H
#define UCC_TEST_HELPERS_H

#include "gtest.h"

#ifndef UINT16_MAX
#define UINT16_MAX (65535)
#endif /* UINT16_MAX */


/* Test output */
#define UCC_TEST_MESSAGE \
    ucc::detail::message_stream("INFO")


/* Skip test */
#define UCC_TEST_SKIP \
    do { \
        throw ucc::test_skip_exception(); \
    } while(0)


#define UCC_TEST_SKIP_R(_reason) \
    do { \
        throw ucc::test_skip_exception(_reason); \
    } while(0)


/* Abort test */
#define UCC_TEST_ABORT(_message) \
    do { \
        std::stringstream ss; \
        ss << _message; \
        GTEST_MESSAGE_(ss.str().c_str(), ::testing::TestPartResult::kFatalFailure); \
        throw ucc::test_abort_exception(); \
    } while(0)


/* UCS error check */
#define EXPECT_UCC_OK(_expr) \
    do { \
        ucs_status_t _status = (_expr); \
        EXPECT_EQ(UCC_OK, _status) << "Error: " << ucs_status_string(_status); \
    } while (0)


#define ASSERT_UCC_OK(_expr, ...) \
    do { \
        ucs_status_t _status = (_expr); \
        if ((_status) != UCC_OK) { \
            UCC_TEST_ABORT("Error: " << ucs_status_string(_status)  __VA_ARGS__); \
        } \
    } while (0)

namespace ucc {

extern std::set< const ::testing::TestInfo*> skipped_tests;

class test_abort_exception : public std::exception {
};


class exit_exception : public std::exception {
public:
    exit_exception(bool failed) : m_failed(failed) {
    }

    virtual ~exit_exception() throw() {
    }

    bool failed() const {
        return m_failed;
    }

private:
    const bool m_failed;
};


class test_skip_exception : public std::exception {
public:
    test_skip_exception(const std::string& reason = "") : m_reason(reason) {
    }
    virtual ~test_skip_exception() throw() {
    }

    virtual const char* what() const throw() {
        return m_reason.c_str();
    }

private:
    const std::string m_reason;
};

class size_value {
public:
    explicit size_value(size_t value) : m_value(value) {}

    size_t value() const {
        return m_value;
    }
private:
    size_t m_value;
};


template <typename O>
static inline O& operator<<(O& os, const size_value& sz)
{
    size_t v = sz.value();

    std::iostream::fmtflags f(os.flags());

    /* coverity[format_changed] */
    os << std::fixed << std::setprecision(1);
    if (v < 1024) {
        os << v;
    } else if (v < 1024 * 1024) {
        os << (v / 1024.0) << "k";
    } else if (v < 1024 * 1024 * 1024) {
        os << (v / 1024.0 / 1024.0) << "m";
    } else {
        os << (v / 1024.0 / 1024.0 / 1024.0) << "g";
    }

    os.flags(f);
    return os;
}

template <typename T>
std::string to_string(const T& value) {
    std::stringstream ss;
    ss << value;
    return ss.str();
}

template <typename T>
std::string to_hex_string(const T& value) {
    std::stringstream ss;
    ss << std::hex << value;
    return ss.str();
}

template <typename T>
T from_string(const std::string& str) {
    T value;
    return (std::stringstream(str) >> value).fail() ? 0 : value;
}

namespace detail {

#define ucc_max(_a, _b) ((_a > _b) ? _a : _b) // TODO: remove after ucc_max is merged upstream correctly

class message_stream {
public:
    message_stream(const std::string& title) {
        static const char PADDING[] = "          ";
        static const size_t WIDTH = strlen(PADDING);

        msg <<  "[";
        msg.write(PADDING, ucc_max(WIDTH - 1, title.length()) - title.length());
        msg << title << " ] ";
    }

    ~message_stream() {
        msg << std::endl;
        std::cout << msg.str() << std::flush;
    }

    template <typename T>
    message_stream& operator<<(const T& value) {
        msg << value;
        return *this;
    }

    message_stream& operator<< (std::ostream&(*f)(std::ostream&)) {
        if (f == (std::basic_ostream<char>& (*)(std::basic_ostream<char>&)) &std::flush) {
            std::string s = msg.str();
            if (!s.empty()) {
                std::cout << s << std::flush;
                msg.str("");
            }
            msg.clear();
        } else {
            msg << f;
        }
        return *this;
    }

    message_stream& operator<< (const size_value& value) {
            msg << value.value();
            return *this;
    }

    std::iostream::fmtflags flags() {
        return msg.flags();
    }

    void flags(std::iostream::fmtflags f) {
        msg.flags(f);
    }
private:
    std::ostringstream msg;
};

} // detail

} // ucc

#endif /* UCC_TEST_HELPERS_H */
