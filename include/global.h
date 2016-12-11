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

struct TEMPO {
	float ida;
	float principal;
	float volta;
	float total;

	TEMPO(): ida(0.0), principal(0.0), volta(0.0), total(0.0) {}

	void set_ida(float t) {
		total += (ida = t);
	}

	void set_principal(float t) {
		total += (principal = t);
	}

	void set_volta(float t) {
		total += (volta = t);
	}
};

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