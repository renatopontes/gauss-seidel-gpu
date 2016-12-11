#ifndef _GLOBAL_H
#define _GLOBAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "clock_timer.h"

#define PAR 0
#define IMPAR 1

#define FIXO 0
#define LOCAL 1

#define SHOW_ERR(msg) { fprintf(stderr, "ERROR: %s\n", msg); \
exit(-1); }

#define sq(x) ((x)*(x))

extern float *malha;

extern int n1, n2;
extern float un, ue, us, uo;
extern float h1;
extern float h2;
extern const float w_fixo;
extern const float pi;

__host__ __device__ float get_a(float x, float y);
__host__ __device__ float get_b(float x, float y);

#endif