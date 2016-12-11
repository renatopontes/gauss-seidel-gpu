#ifndef _GLOBAL_H
#define _GLOBAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define SHOW_ERR(msg) { fprintf(stderr, "ERROR: %s\n", msg); \
exit(-1); }

#define sq(x) ((x)*(x))

extern int n1, n2;
extern float un, ue, us, uo;
extern float h1;
extern float h2;
extern float *malha;
extern const float w_fixo;
extern const float PI;

typedef struct _COORD_MALHA {
	float xi, yj;

	_COORD_MALHA(float _xi, float _yj): xi(_xi), yj(_yj) {}
} COORD_MALHA;

COORD_MALHA valor(int i, int j);
float get_a(float x, float y);
float get_b(float x, float y);
float get_v(int i, int j);

#endif