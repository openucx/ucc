/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2014. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "log.h"

#include <utils/ucc_sys.h>
#include <utils/ucc_string.h>
#include <utils/ucc_spinlock.h>
#include <utils/ucc_atomic.h>
#include <utils/ucc_assert.h>
#include <utils/ucc_malloc.h>
#include <fnmatch.h>


#define UCC_MAX_LOG_HANDLERS    32

#define UCC_LOG_TIME_FMT        "[%lu.%06lu]"
#define UCC_LOG_METADATA_FMT    "%17s:%-4u %-4s %-5s %*s"
#define UCC_LOG_PROC_DATA_FMT   "[%s:%-5d:%s]"

#define UCC_LOG_COMPACT_FMT     UCC_LOG_TIME_FMT " " UCC_LOG_PROC_DATA_FMT "  "
#define UCC_LOG_SHORT_FMT       UCC_LOG_TIME_FMT " [%s] " UCC_LOG_METADATA_FMT "%s\n"
#define UCC_LOG_FMT             UCC_LOG_TIME_FMT " " UCC_LOG_PROC_DATA_FMT " " \
                                UCC_LOG_METADATA_FMT "%s\n"

#define UCC_LOG_TIME_ARG(_tv)  (_tv)->tv_sec, (_tv)->tv_usec

#define UCC_LOG_METADATA_ARG(_short_file, _line, _level, _comp_conf) \
    (_short_file), (_line), (_comp_conf)->name, \
    ucc_log_level_names[_level], (ucc_log_current_indent * 2), ""

#define UCC_LOG_PROC_DATA_ARG() \
    ucc_log_hostname, ucc_log_get_pid(), ucc_log_get_thread_name()

#define UCC_LOG_COMPACT_ARG(_tv)\
    UCC_LOG_TIME_ARG(_tv), UCC_LOG_PROC_DATA_ARG()

#define UCC_LOG_SHORT_ARG(_short_file, _line, _level, _comp_conf, _tv, \
                          _message) \
    UCC_LOG_TIME_ARG(_tv), ucc_log_get_thread_name(), \
            UCC_LOG_METADATA_ARG(_short_file, _line, _level, _comp_conf), \
            (_message)

#define UCC_LOG_ARG(_short_file, _line, _level, _comp_conf, _tv, _message) \
    UCC_LOG_TIME_ARG(_tv), UCC_LOG_PROC_DATA_ARG(), \
    UCC_LOG_METADATA_ARG(_short_file, _line, _level, _comp_conf), (_message)

KHASH_MAP_INIT_STR(ucc_log_filter, char);

const char *ucc_log_level_names[] = {
    [UCC_LOG_LEVEL_FATAL]        = "FATAL",
    [UCC_LOG_LEVEL_ERROR]        = "ERROR",
    [UCC_LOG_LEVEL_WARN]         = "WARN",
    [UCC_LOG_LEVEL_DIAG]         = "DIAG",
    [UCC_LOG_LEVEL_INFO]         = "INFO",
    [UCC_LOG_LEVEL_DEBUG]        = "DEBUG",
    [UCC_LOG_LEVEL_TRACE]        = "TRACE",
    [UCC_LOG_LEVEL_TRACE_REQ]    = "REQ",
    [UCC_LOG_LEVEL_TRACE_DATA]   = "DATA",
    [UCC_LOG_LEVEL_TRACE_ASYNC]  = "ASYNC",
    [UCC_LOG_LEVEL_TRACE_FUNC]   = "FUNC",
    [UCC_LOG_LEVEL_TRACE_POLL]   = "POLL",
    [UCC_LOG_LEVEL_LAST]         = NULL,
    [UCC_LOG_LEVEL_PRINT]        = "PRINT"
};

static unsigned ucc_log_handlers_count       = 0;
static int ucc_log_initialized               = 0;
static int __thread ucc_log_current_indent   = 0;
static char ucc_log_hostname[HOST_NAME_MAX]  = {0};
static int ucc_log_pid                       = 0;
static FILE *ucc_log_file                    = NULL;
static char *ucc_log_file_base_name          = NULL;
static int ucc_log_file_close                = 0;
static int ucc_log_file_last_idx             = 0;
static uint32_t ucc_log_thread_count         = 0;
static char __thread ucc_log_thread_name[32] = {0};
static ucc_log_func_t ucc_log_handlers[UCC_MAX_LOG_HANDLERS];
static ucc_spinlock_t ucc_log_global_filter_lock;
static khash_t(ucc_log_filter) ucc_log_global_filter;

static inline int ucc_log_get_pid()
{
    if (ucc_unlikely(ucc_log_pid == 0)) {
        return getpid();
    }

    return ucc_log_pid;
}

static const char *ucc_log_get_thread_name()
{
    char *name = ucc_log_thread_name;
    uint32_t thread_num;

    if (ucc_unlikely(name[0] == '\0')) {
        thread_num = ucc_atomic_fadd32(&ucc_log_thread_count, 1);
        ucc_snprintf_safe(ucc_log_thread_name, sizeof(ucc_log_thread_name),
                          "%u", thread_num);
    }

    return name;
}

void ucc_log_flush()
{
    if (ucc_log_file != NULL) {
        fflush(ucc_log_file);

        if (ucc_log_file_close) { /* non-stdout/stderr */
            fsync(fileno(ucc_log_file));
        }
    }
}

size_t ucc_log_get_buffer_size()
{
    return ucc_config_memunits_get(ucc_global_config.log_buffer_size,
                                   256, UCC_ALLOCA_MAX_SIZE);
}

static void ucc_log_get_file_name(char *log_file_name, size_t max, int idx)
{
    ucc_assert(idx <= ucc_global_config.log_file_rotate);

    if (idx == 0) {
        ucc_strncpy_zero(log_file_name, ucc_log_file_base_name, max);
        return;
    }

    ucc_snprintf_zero(log_file_name, max, "%s.%d",
                      ucc_log_file_base_name, idx);
}

static void ucc_log_file_rotate()
{
    char *old_log_file_name, *new_log_file_name;
    int idx, ret;
    ucc_status_t status;

    status = ucc_string_alloc_path_buffer(&old_log_file_name,
                                          "old_log_file_name");
    if (status != UCC_OK) {
        goto out;
    }

    if (ucc_log_file_last_idx == ucc_global_config.log_file_rotate) {
        /* remove the last file and log rotation from the
         * `log_file_rotate - 1` file */
        ucc_log_get_file_name(old_log_file_name, PATH_MAX,
                              ucc_log_file_last_idx);
        unlink(old_log_file_name);
    } else {
        ucc_log_file_last_idx++;
    }

    ucc_assert(ucc_log_file_last_idx <= ucc_global_config.log_file_rotate);

    status = ucc_string_alloc_path_buffer(&new_log_file_name,
                                          "new_log_file_name");
    if (status != UCC_OK) {
        goto out_free_old_log_file_name;
    }

    for (idx = ucc_log_file_last_idx - 1; idx >= 0; --idx) {
        ucc_log_get_file_name(old_log_file_name, PATH_MAX, idx);
        ucc_log_get_file_name(new_log_file_name, PATH_MAX, idx + 1);

        if (access(old_log_file_name, W_OK) != 0) {
            ucc_fatal("unable to write to %s", old_log_file_name);
        }

        /* coverity[toctou] */
        ret = rename(old_log_file_name, new_log_file_name);
        if (ret) {
            ucc_fatal("failed to rename %s to %s: %m",
                      old_log_file_name, new_log_file_name);
        }


        if (access(old_log_file_name, F_OK) != -1) {
            ucc_fatal("%s must not exist on the filesystem", old_log_file_name);
        }

        if (access(new_log_file_name, W_OK) != 0) {
            ucc_fatal("unable to write to %s", new_log_file_name);
        }
    }

    ucc_free(new_log_file_name);
out_free_old_log_file_name:
    ucc_free(old_log_file_name);
out:
    return;
}

static void ucc_log_handle_file_max_size(int log_entry_len)
{
    const char *next_token;

    /* check if it is necessary to find a new storage for logs */
    if ((log_entry_len + ftell(ucc_log_file)) < ucc_global_config.log_file_size) {
        return;
    }

    fclose(ucc_log_file);

    if (ucc_global_config.log_file_rotate != 0) {
        ucc_log_file_rotate();
    } else {
        unlink(ucc_log_file_base_name);
    }

    ucc_open_output_stream(ucc_log_file_base_name, UCC_LOG_LEVEL_FATAL,
                           &ucc_log_file, &ucc_log_file_close,
                           &next_token, NULL);
}

void ucc_log_print_compact(const char *str)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);

    if (ucc_log_initialized) {
        if (ucc_log_file_close) { /* non-stdout/stderr */
            ucc_log_handle_file_max_size(strlen(str) + 1);
        }

        fprintf(ucc_log_file, UCC_LOG_COMPACT_FMT " %s\n",
                UCC_LOG_COMPACT_ARG(&tv), str);
    } else {
        fprintf(stdout, UCC_LOG_COMPACT_FMT " %s\n", UCC_LOG_COMPACT_ARG(&tv),
                str);
    }
}

static void ucc_log_print(const char *short_file, int line,
                          ucc_log_level_t level,
                          const ucc_log_component_config_t *comp_conf,
                          const struct timeval *tv, const char *message)
{
    int log_entry_len;

    if (ucc_log_initialized) {
        if (ucc_log_file_close) { /* non-stdout/stderr */
            /* get log entry size */
            log_entry_len = snprintf(NULL, 0, UCC_LOG_FMT,
                                     UCC_LOG_ARG(short_file, line, level,
                                                 comp_conf, tv, message));
            ucc_log_handle_file_max_size(log_entry_len);
        }

        fprintf(ucc_log_file, UCC_LOG_FMT,
                UCC_LOG_ARG(short_file, line, level,
                            comp_conf, tv, message));
    } else {
        fprintf(stdout, UCC_LOG_SHORT_FMT,
                UCC_LOG_SHORT_ARG(short_file, line, level,
                                  comp_conf, tv, message));
    }
}

ucc_log_func_rc_t
ucc_log_default_handler(const char *file, unsigned line, const char *function,
                        ucc_log_level_t level,
                        const ucc_log_component_config_t *comp_conf,
                        const char *format, va_list ap)
{
    size_t buffer_size = ucc_log_get_buffer_size();
    char *saveptr      = "";
    const char *short_file;
    struct timeval tv;
    khiter_t khiter;
    char *log_line;
    char match;
    int khret;
    char *buf;
    const char *filename;

    if (!ucc_log_component_is_enabled(level, comp_conf) &&
        (level != UCC_LOG_LEVEL_PRINT)) {
        return UCC_LOG_FUNC_RC_CONTINUE;
    }

    ucc_spin_lock(&ucc_log_global_filter_lock);
    khiter = kh_get(ucc_log_filter, &ucc_log_global_filter, file);
    if (ucc_unlikely(khiter == kh_end(&ucc_log_global_filter))) {
        /* Add source file name to the hash */
        match  = fnmatch(ucc_global_config.log_component.file_filter, file, 0) !=
                 FNM_NOMATCH;

        filename = ucc_strdup(file, "log filter filename");
        if (filename == NULL) {
            ucc_fatal("cannot allocate log filtering entry for '%s'", file);
        }

        khiter = kh_put(ucc_log_filter, &ucc_log_global_filter, filename,
                        &khret);
        ucc_assert((khret == UCS_KH_PUT_BUCKET_EMPTY) ||
                   (khret == UCS_KH_PUT_BUCKET_CLEAR));
        kh_val(&ucc_log_global_filter, khiter) = match;
    } else {
        match = kh_val(&ucc_log_global_filter, khiter);
    }
    ucc_spin_unlock(&ucc_log_global_filter_lock);

    if (!match) {
        return UCC_LOG_FUNC_RC_CONTINUE;
    }

    buf = ucc_alloca(buffer_size + 1);
    buf[buffer_size] = 0;
    vsnprintf(buf, buffer_size, format, ap);

    if (level <= ucc_global_config.log_level_trigger) {
        ucc_fatal_error_message(file, line, function, buf);
    } else {
        short_file = ucc_basename(file);
        gettimeofday(&tv, NULL);

        log_line = strtok_r(buf, "\n", &saveptr);
        while (log_line != NULL) {
            ucc_log_print(short_file, line, level, comp_conf, &tv, log_line);
            log_line = strtok_r(NULL, "\n", &saveptr);
        }
    }

    /* flush the log file if the log_level of this message is fatal or error */
    if (level <= UCC_LOG_LEVEL_ERROR) {
        ucc_log_flush();
    }

    return UCC_LOG_FUNC_RC_CONTINUE;
}

void ucc_log_push_handler(ucc_log_func_t handler)
{
    if (ucc_log_handlers_count < UCC_MAX_LOG_HANDLERS) {
        ucc_log_handlers[ucc_log_handlers_count++] = handler;
    }
}

void ucc_log_pop_handler()
{
    if (ucc_log_handlers_count > 0) {
        --ucc_log_handlers_count;
    }
}

void ucc_log_indent(int delta)
{
    ucc_log_current_indent += delta;
    ucc_assert(ucc_log_current_indent >= 0);
}

int ucc_log_get_current_indent()
{
    return ucc_log_current_indent;
}

unsigned ucc_log_num_handlers()
{
    return ucc_log_handlers_count;
}

void ucc_log_dispatch(const char *file, unsigned line, const char *function,
                      ucc_log_level_t level, ucc_log_component_config_t *comp_conf,
                      const char *format, ...)
{
    ucc_log_func_rc_t rc;
    unsigned idx;
    va_list ap;

    /* Call handlers in reverse order */
    rc    = UCC_LOG_FUNC_RC_CONTINUE;
    idx = ucc_log_handlers_count;
    while ((idx > 0) && (rc == UCC_LOG_FUNC_RC_CONTINUE)) {
        --idx;
        va_start(ap, format);
        rc = ucc_log_handlers[idx](file, line, function,
                                   level, comp_conf, format, ap);
        va_end(ap);
    }
}

void ucc_log_fatal_error(const char *format, ...)
{
    size_t buffer_size = ucc_log_get_buffer_size();
    FILE *stream = stderr;
    char *buffer, *p;
    va_list ap;
    int ret;

    /* Initialize va_list before any potential analyzer-confusing macros */
    va_start(ap, format);

    buffer = ucc_alloca(buffer_size + 1);
    p = buffer;

    /* Print hostname:pid */
    snprintf(p, buffer_size, "[%s:%-5d:%s:%d] ", ucc_log_hostname,
             ucc_log_get_pid(), ucc_log_get_thread_name(), ucc_get_tid());
    buffer_size -= strlen(p);
    p           += strlen(p);

    /* Print rest of the message */
    // NOLINTNEXTLINE(clang-analyzer-valist.Uninitialized)
    vsnprintf(p, buffer_size, format, ap);
    va_end(ap);
    buffer_size -= strlen(p);
    p           += strlen(p);

    /* Newline */
    snprintf(p, buffer_size, "\n");

    /* Flush stderr, and write the message directly to the pipe */
    fflush(stream);
    ret = write(fileno(stream), buffer, strlen(buffer));
    (void)ret;
}

/**
 * Print a bitmap as a list of ranges.
 *
 * @param n        Number equivalent to the first bit in the bitmap.
 * @param bitmap   Compressed array of bits.
 * @param length   Number of bits in the bitmap.
 */
const char *ucc_log_bitmap_to_str(unsigned n, uint8_t *bitmap, size_t length)
{
    static char buf[512] = {0};
    int first, in_range;
    unsigned prev = 0, end = 0;
    char *p, *endp;
    size_t i;

    p = buf;
    endp = buf + sizeof(buf) - 4;

    first = 1;
    in_range = 0;
    for (i = 0; i < length; ++i) {
        if (bitmap[i / 8] & UCC_BIT(i % 8)) {
            if (first) {
                p += snprintf(p, endp - p, "%d", n);
                if (p > endp) {
                    goto overflow;
                }
            } else if (n == prev + 1) {
                in_range = 1;
                end = n;
            } else {
                if (in_range) {
                    p += snprintf(p, endp - p, "-%d", end);
                    if (p > endp) {
                        goto overflow;
                    }
                }
                in_range = 0;
                p += snprintf(p, endp - p, ",%d", n);
                if (p > endp) {
                    goto overflow;
                }
            }
            first = 0;
            prev = n;
        }
        ++n;
    }
    if (in_range) {
        p += snprintf(p, endp - p, "-%d", end);
        if (p > endp) {
            goto overflow;
        }
    }
    return buf;

overflow:
    {
        /* Ensure we do not overflow the buffer when indicating truncation */
        size_t remaining = (size_t)((buf + sizeof(buf) - 1) - p);
        if (remaining > 0) {
            ucc_strncpy_zero(p, "...", remaining + 1);
        }
    }
    return buf;
}

void ucc_log_early_init()
{
    ucc_log_initialized   = 0;
    ucc_log_hostname[0]   = 0;
    ucc_log_pid           = getpid();
    ucc_log_file          = NULL;
    ucc_log_file_last_idx = 0;
    ucc_log_file_close    = 0;
    ucc_log_thread_count  = 0;
}

static void ucc_log_atfork_prepare()
{
    ucc_log_pid = 0;
    ucc_log_flush();
}

static void ucc_log_atfork_post()
{
    ucc_log_pid = getpid();
}

void ucc_log_init()
{
    const char *next_token;

    if (ucc_log_initialized) {
        return;
    }

    ucc_log_initialized = 1; /* Set this to 1 immediately to avoid infinite recursion */

    if (ucc_global_config.log_file_size < ucc_log_get_buffer_size()) {
        ucc_fatal("the maximal log file size (%zu) has to be >= %zu",
                  ucc_global_config.log_file_size,
                  ucc_log_get_buffer_size());
    }

    if (ucc_global_config.log_file_rotate > INT_MAX) {
        ucc_fatal("the log file rotate (%u) has to be <= %d",
                  ucc_global_config.log_file_rotate, INT_MAX);
    }

    ucc_spinlock_init(&ucc_log_global_filter_lock, 0);
    kh_init_inplace(ucc_log_filter, &ucc_log_global_filter);

    ucc_strncpy_zero(ucc_log_hostname, ucc_get_host_name(), sizeof(ucc_log_hostname));
    ucc_log_file           = stdout;
    ucc_log_file_base_name = NULL;
    ucc_log_file_close     = 0;
    ucc_log_file_last_idx  = 0;

    ucc_log_push_handler(ucc_log_default_handler);

    if (strlen(ucc_global_config.log_file) != 0) {
        ucc_open_output_stream(ucc_global_config.log_file, UCC_LOG_LEVEL_FATAL,
                               &ucc_log_file, &ucc_log_file_close,
                               &next_token, &ucc_log_file_base_name);
    }

    pthread_atfork(ucc_log_atfork_prepare, ucc_log_atfork_post,
                   ucc_log_atfork_post);
}

void ucc_log_cleanup()
{
    const char *filename;

    ucc_assert(ucc_log_initialized);

    ucc_log_flush();
    if (ucc_log_file_close) {
        fclose(ucc_log_file);
    }

    ucc_spinlock_destroy(&ucc_log_global_filter_lock);

    kh_foreach_key(&ucc_log_global_filter, filename,
                   { ucc_free((void*)filename); })
        ; /* code format script wants it there */

    kh_destroy_inplace(ucc_log_filter, &ucc_log_global_filter);

    ucc_free(ucc_log_file_base_name);
    ucc_log_file_base_name = NULL;
    ucc_log_file           = NULL;
    ucc_log_file_last_idx  = 0;
    ucc_log_initialized    = 0;
    ucc_log_handlers_count = 0;
}

void ucc_log_print_backtrace(ucc_log_level_t level)
{
    (void)level; /* suppress unused parameter warning */
    //TODO: fix this
    ucc_assert_always(0);
}

void ucc_log_set_thread_name(const char *format, ...)
{
    va_list ap;

    va_start(ap, format);
    memset(ucc_log_thread_name, 0, sizeof(ucc_log_thread_name));
    // NOLINTNEXTLINE(clang-analyzer-valist.Uninitialized)
    vsnprintf(ucc_log_thread_name, sizeof(ucc_log_thread_name) - 1, format, ap);
    va_end(ap);
}
