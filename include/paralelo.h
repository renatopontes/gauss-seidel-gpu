#ifndef _PARALELO
#define _PARALELO

#include "global.h"

#define TAM_BLOCO 8

#define CUDA_SAFE_CALL(call) { \
    cudaError_t err = call;     \
    if(err != cudaSuccess) {    \
        fprintf(stderr,"Erro no arquivo '%s', linha %i: %s.\n", \
            __FILE__, __LINE__,cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); } }

typedef struct {
	int n1, n2;
	float un, ue, us, uo;
	float h1, h2;
	float w_fixo;
	float pi;
} GLOBALS;

// Método de Gauss-Seidel paralelizado com sobre-relaxação sucessiva
// com iter iterações. Usa w variável se modo == LOCAL.
void gauss_seidel_par(int iter, int modo=FIXO);

#endif