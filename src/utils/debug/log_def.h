/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCC_LOG_DEF_H_
#define UCC_LOG_DEF_H_

#ifndef UCC_MAX_LOG_LEVEL
#  define UCC_MAX_LOG_LEVEL  UCC_LOG_LEVEL_LAST
#endif

#include <utils/debug/types.h>
#include <utils/ucc_compiler_def.h>
#include <core/ucc_global_opts.h>
#include <stdarg.h>
#include <stdint.h>


BEGIN_C_DECLS

/** @file log_def.h */

#define ucc_log_component_is_enabled(_level, _comp_log_config) \
    ucc_unlikely(((_level) <= UCC_MAX_LOG_LEVEL) && \
                 ((_level) <= (((ucc_log_component_config_t*)(_comp_log_config))->log_level)))

#define ucc_log_is_enabled(_level) \
    ucc_log_component_is_enabled(_level, &ucc_global_config.log_component)

#define ucc_log_component(_level, _comp_log_config, _fmt, ...) \
    do { \
        if (ucc_log_component_is_enabled(_level, _comp_log_config)) { \
            ucc_log_dispatch(__FILE__, __LINE__, __func__, \
                             (ucc_log_level_t)(_level), _comp_log_config, _fmt, ## __VA_ARGS__); \
        } \
    } while (0)

#define ucc_log(_level, _fmt, ...) \
    do { \
        ucc_log_component(_level, &ucc_global_config.log_component, _fmt, ## __VA_ARGS__); \
    } while (0)

#define ucc_error(_fmt, ...)        ucc_log(UCC_LOG_LEVEL_ERROR, _fmt, ## __VA_ARGS__)
#define ucc_warn(_fmt, ...)         ucc_log(UCC_LOG_LEVEL_WARN, _fmt,  ## __VA_ARGS__)
#define ucc_diag(_fmt, ...)         ucc_log(UCC_LOG_LEVEL_DIAG, _fmt,  ## __VA_ARGS__)
#define ucc_info(_fmt, ...)         ucc_log(UCC_LOG_LEVEL_INFO, _fmt, ## __VA_ARGS__)
#define ucc_debug(_fmt, ...)        ucc_log(UCC_LOG_LEVEL_DEBUG, _fmt, ##  __VA_ARGS__)
#define ucc_trace(_fmt, ...)        ucc_log(UCC_LOG_LEVEL_TRACE, _fmt, ## __VA_ARGS__)
#define ucc_trace_req(_fmt, ...)    ucc_log(UCC_LOG_LEVEL_TRACE_REQ, _fmt, ## __VA_ARGS__)
#define ucc_trace_data(_fmt, ...)   ucc_log(UCC_LOG_LEVEL_TRACE_DATA, _fmt, ## __VA_ARGS__)
#define ucc_trace_async(_fmt, ...)  ucc_log(UCC_LOG_LEVEL_TRACE_ASYNC, _fmt, ## __VA_ARGS__)
#define ucc_trace_func(_fmt, ...)   ucc_log(UCC_LOG_LEVEL_TRACE_FUNC, "%s(" _fmt ")", __func__, ## __VA_ARGS__)
#define ucc_trace_poll(_fmt, ...)   ucc_log(UCC_LOG_LEVEL_TRACE_POLL, _fmt, ## __VA_ARGS__)

#define ucc_log_indent_level(_level, _delta) \
    do { \
        if (ucc_log_component_is_enabled(_level, \
                                         &ucc_global_config.log_component)) { \
            ucc_log_indent(_delta); \
        } \
    } while (0)


/**
 * Print a message regardless of current log level. Output can be
 * enabled/disabled via environment variable/configuration settings.
 *
 * During debugging it can be useful to add a few prints to the code
 * without changing a current log level. Also it is useful to be able
 * to see messages only from specific processes. For example, one may
 * want to see prints only from rank 0 when debugging MPI.
 *
 * The function is intended for debugging only. It should not be used
 * in the real code.
 */

#define ucc_print(_fmt, ...) \
    do { \
        if (ucc_global_config.log_print_enable) { \
            ucc_log_dispatch(__FILE__, __LINE__, __func__, \
                             UCC_LOG_LEVEL_PRINT, &ucc_global_config.log_component, _fmt, ## __VA_ARGS__); \
        } \
    } while(0)


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


extern const char *ucc_log_level_names[];

/**
 * Dispatch a logging message.
 *
 * @param [in] file       Source file name.
 * @param [in] line       Source line number.
 * @param [in] function   Function name which generated the log.
 * @param [in] level      Log level of the message.
 * @param [in] comp_conf  Component log config.
 * @param [in] message    Log format.
 */
void ucc_log_dispatch(const char *file, unsigned line, const char *function,
                      ucc_log_level_t level, ucc_log_component_config_t *comp_conf,
                      const char *format, ...)
    UCC_F_PRINTF(6, 7);


/**
 * Flush logging output.
 */
void ucc_log_flush(void);


/**
 * @return Configured log buffer size
 */
size_t ucc_log_get_buffer_size(void);


/**
 * Print a compact log line (without file/line prefixes) to the log stream.
 *
 * @param [in] str   Log line to print.
 */
void ucc_log_print_compact(const char *str);


/**
 * Default log handler, which prints the message to the output configured in
 * UCC global options. See @ref ucc_log_func_t.
 */
ucc_log_func_rc_t
ucc_log_default_handler(const char *file, unsigned line, const char *function,
                        ucc_log_level_t level,
                        const ucc_log_component_config_t *comp_conf,
                        const char *format, va_list ap);


/**
 * Show a fatal error
 */
void ucc_log_fatal_error(const char *format, ...) UCC_F_PRINTF(1, 2);


/**
 * Initialize/cleanup logging subsystem.
 */
void ucc_log_early_init(void);
void ucc_log_init(void);
void ucc_log_cleanup(void);


const char *ucc_log_bitmap_to_str(unsigned n, uint8_t *bitmap, size_t length);

/**
 * Add/remove logging handlers
 */
void ucc_log_push_handler(ucc_log_func_t handler);
void ucc_log_pop_handler(void);
unsigned ucc_log_num_handlers(void);


/**
 * Add indentation to all subsequent log messages.
 *
 * @param [in] delta   How much indentation to add, on top of the current
 *                     indentation level.
 *                     A negative number will reduce the indentation level.
 */
void ucc_log_indent(int delta);


/**
 * @return Current log indent level.
 */
int ucc_log_get_current_indent(void);


/**
 * Log backtrace.
 *
 * @param level          Log level.
 */
void ucc_log_print_backtrace(ucc_log_level_t level);


/**
 * Set the name for current thread, to appear in log messages
 *
 * @param name           Thread name to set
 */
void ucc_log_set_thread_name(const char *format, ...) UCC_F_PRINTF(1, 2);

END_C_DECLS

#endif
