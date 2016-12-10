/* Alunos: 																	*/
/*			Renato Pontes Rodrigues                                         */
/*			Mateus Ildefonso do Nascimento									*/

/* Para compilar:                                                           */
/* nvcc matrix_mult.cu -o matrix_mult                                       */
/* com verificação: -DCHECK                                                 */
/* com impressão das matrizes: -DPRINT_M                                    */
/* impressão pra facilitar fazer a planilha: -DNOT_VERBOSE                  */
/* DEFINIR TAMANHO DOS BLOCOS (OBRIGATÓRIO): -DTAM_BLOCO=tam                */
/* VERSÃO COM MEMORIA COMPARTILHADA: -DSHARED                               */
/* PS: tem um arquivo roda_testes que só roda no Windows mas lá dá pra ver  */
/* exemplos de compilação.                                                  */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "clock_timer.h"

#define sq(x) ((x)*(x))

int n1, n2;
float un = 5, ue = 10, us = 5, uo = 0;
float h1;
float h2;

typedef struct _COORD_MALHA {
	float xi, yj;

	_COORD_MALHA(float _xi, float _yj): xi(_xi), yj(_yj) {}
} COORD_MALHA;

COORD_MALHA valor(int i, int j) {
	float xi, yj;

	xi = (i+1)*h1;
	yj = (j+1)*h2;

	return COORD_MALHA(xi, yj);
}

void init_malha(float *malha) {
	srand(time(NULL));
	for (int i = 0; i < n1*n2; ++i) {
		malha[i] = (1.0 * rand() / RAND_MAX) * 10.0;
	}
}

float a(float x, float y) {
	return 500.0 * x * (1.0 - x) * (0.5 - y);
}

float b(float x, float y) {
	return 500.0 * y * (1.0 - y) * (x - 0.5);
}

float get_v(int i, int j, float *malha) {
	if (i < 0) return uo;
	if (i == n1) return ue;
	if (j < 0) return us;
	if (j == n2) return un;

	return malha[i*n2 + j];
}

void calcula_v(int i, int j, float *malha) {
	COORD_MALHA pos = valor(i, j);
	float o, e, s, n;

	o = (2.0 + h1 * a(pos.xi, pos.yj)) / (4.0 * (1.0 + sq(h1)/sq(h2)));
	e = (2.0 - h1 * a(pos.xi, pos.yj)) / (4.0 * (1.0 + sq(h1)/sq(h2)));
	s = (2.0 + h2 * b(pos.xi, pos.yj)) / (4.0 * (1.0 + sq(h2)/sq(h1)));
	n = (2.0 - h2 * b(pos.xi, pos.yj)) / (4.0 * (1.0 + sq(h2)/sq(h1)));

	malha[i*n2 + j] = o*get_v(i-1,j, malha) + e*get_v(i+1, j, malha) + s*get_v(i, j-1, malha) + n*get_v(i, j+1, malha);
}

void processa_malha(float *malha) {
	for (int i = 0; i < n1; ++i) {
		for (int j = i % 2; j < n2; j += 2) {
			calcula_v(i, j, malha);
		}
	}

	for (int i = 0; i < n1; ++i) {
		for (int j = (i+1) % 2; j < n2; j += 2) {
			calcula_v(i, j, malha);
		}
	}
}

void gauss_seidel(float *malha, int iter) {
	while(iter--) {
		processa_malha(malha);
	}
}

int main(int argc, char **argv) {
	float *malha;

	n1 = atoi(argv[1]);
	n2 = atoi(argv[2]);

	h1 = 1.0 / (n1 + 1);
	h2 = 1.0 / (n2 + 1);

	malha = (float *) malloc(n1 * n2 * sizeof(float));

	init_malha(malha);
	gauss_seidel(malha, 5000);

	for(int j = n2; j >= -1; --j) {
		for (int i = -1; i <= n1 ; ++i) {
			printf("%f ", get_v(i, j, malha));
		}
		printf("\n");
	}

	// printf("\n");
	// for(int i = -1; i <= n1; ++i) {
	// 	for (int j = -1; j <= n2 ; ++j) {
	// 		printf("%f ", malha[i*n2 + j]);
	// 	}
	// 	printf("\n");
	// }

	return 0;
}