/* Alunos: 																				*/
/*			Renato Pontes Rodrigues														*/
/*			Mateus Ildefonso do Nascimento												*/

#ifndef _PARALELO
#define _PARALELO

#include "global.h"

#define CUDA_SAFE_CALL(call) { \
    cudaError_t err = call;     \
    if(err != cudaSuccess) {    \
        fprintf(stderr,"Erro no arquivo '%s', linha %i: %s.\n", \
            __FILE__, __LINE__,cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); } }

// Estrutura que reúne os parâmetros globais do problema, para facilitar
// a passagem deles para a memória do device.
typedef struct {
	int n1, n2;
	float un, ue, us, uo;
	float h1, h2;
	float w_fixo;
	float pi;
} GLOBALS;

// Método de Gauss-Seidel paralelizado com sobre-relaxação sucessiva
// com iter iterações. Usa w variável se modo == LOCAL.
// Retorna uma estrutura TEMPO com os tempos de execução.
TEMPO gauss_seidel_par(int iter, int modo=FIXO);

#endif