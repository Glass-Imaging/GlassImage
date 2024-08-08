//
// Created by Pranav Garg on 8/6/24.
//

#ifndef GLASSLIBRARYBUILDER_GLS_SYSLOG_HPP
#define GLASSLIBRARYBUILDER_GLS_SYSLOG_HPP

#include <syslog.h>

#undef LOG_EMERG
#undef LOG_ALERT
#undef LOG_CRIT
#undef LOG_ERR
#undef LOG_WARNING
#undef LOG_NOTICE
#undef LOG_INFO
#undef LOG_DEBUG

/** Corresponds to the Android ERROR log priority. */
#define SYSLOG_EMERG 0
/** Corresponds to the Android ERROR log priority. */
#define SYSLOG_ALERT 1
/** Corresponds to the Android ERROR log priority. */
#define SYSLOG_CRIT 2
/** Corresponds to the Android ERROR log priority. */
#define SYSLOG_ERR 3
/** Corresponds to the Android WARN log priority. */
#define SYSLOG_WARNING 4
/** Corresponds to the Android INFO log priority. */
#define SYSLOG_NOTICE 5
/** Corresponds to the Android INFO log priority. */
#define SYSLOG_INFO 6
/** Corresponds to the Android DEBUG log priority. */
#define SYSLOG_DEBUG 7


#endif //GLASSLIBRARYBUILDER_GLS_SYSLOG_HPP
