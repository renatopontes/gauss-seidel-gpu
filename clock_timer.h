/* Example:
 *    #include "clock_timer.h"
 *    . . .
 *    double start, finish, elapsed;
 *    . . .
 *    GET_TIME(start);
 *    . . .
 *    Code to be timed
 *    . . .
 *    GET_TIME(finish);
 *    elapsed = finish - start;
 *    printf("The code to be timed took %e seconds\n", elapsed);
*/

#ifndef _CLOCK_TIMER_H
#define _CLOCK_TIMER_H

#ifdef _WIN32
#include <windows.h>

#define GET_TIME(now) \
 { \
    LARGE_INTEGER frequency; \
    LARGE_INTEGER end; \
    QueryPerformanceFrequency(&frequency); \
    QueryPerformanceCounter(&end); \
    now = (end.QuadPart) / (double)frequency.QuadPart; \
}

#else
#include <sys/time.h>
#define BILLION 1000000000L

/* The argument now should be a double (not a pointer to a double) */
#define GET_TIME(now) { \
   struct timespec time; \
   clock_gettime(CLOCK_MONOTONIC_RAW, &time); \
   now = time.tv_sec + time.tv_nsec/1000000000.0; \
}
#endif
#endif
