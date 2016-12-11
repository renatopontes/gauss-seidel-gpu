/* Alunos: 																				*/
/*			Renato Pontes Rodrigues														*/
/*			Mateus Ildefonso do Nascimento												*/

/* Para compilar:																		*/
/* 			make																		*/

#include "include/global.h"
#include "include/clock_timer.h"
#include "include/sequencial.h"
#include "include/paralelo.h"

int n1, n2;
float un = 5, ue = 10, us = 5, uo = 0;
float h1;
float h2;
float *malha;
const float w_fixo = 0.6;
const float PI = 2.0*acos(0);

COORD_MALHA valor(int i, int j) {
	float xi, yj;

	xi = (i+1)*h1;
	yj = (j+1)*h2;

	return COORD_MALHA(xi, yj);
}

float get_a(float x, float y) {
	return 500.0 * x * (1.0 - x) * (0.5 - y);
}

float get_b(float x, float y) {
	return 500.0 * y * (1.0 - y) * (x - 0.5);
}

float get_v(int i, int j) {
	if (i < 0) return uo;
	if (i == n1) return ue;
	if (j < 0) return us;
	if (j == n2) return un;

	return malha[i*n2 + j];
}

void init_malha() {
	srand(time(NULL));
	for (int i = 0; i < n1*n2; ++i) {
		malha[i] = (1.0 * rand() / RAND_MAX) * 10.0;
	}
}

int main(int argc, char **argv) {
	FILE *fout;
	const int iter = 1000;

	if (argc < 3) {
		SHOW_ERR("Passagem incorreta de parametros.\n\n"
			"\tUso: ./gauss_seidel N1 N2 [sw|sl|pw|pl]\n"
			"\tN1: largura da malha\n"
			"\tN2: altura da malha\n"
			"\tsw: processamento sequencial com sobre-relaxacao sucessiva. (default)\n"
			"\tsl: processamento sequencial com sobre-relaxacao sucessiva local.\n"
			"\tpw: processamento paralelo com sobre-relaxacao sucessiva.\n"
			"\tpl: processamento paralelo com sobre-relaxacao sucessiva local.\n");
	}

	n1 = atoi(argv[1]);
	n2 = atoi(argv[2]);

	h1 = 1.0 / (n1 + 1);
	h2 = 1.0 / (n2 + 1);

	fout = fopen("out/matriz.txt", "w+");
	if (!fout) {
		SHOW_ERR("Nao foi possivel criar arquivo de saida\n");
	}

	malha = (float *) malloc(n1 * n2 * sizeof(float));

	init_malha();

	if (argc == 3 || !strcmp(argv[3], "sw")){
		printf("Processamento sequencial\n"
			"Sobre-relaxacao sucessiva\n");
		gauss_seidel_seq_w(iter);
	}
	else if (!strcmp(argv[3], "sl")){
		printf("Processamento sequencial\n"
			"Sobre-relaxacao sucessiva local\n");
		gauss_seidel_seq_l(iter);
	}

	for(int j = n2; j >= -1; --j) {
		for (int i = -1; i <= n1 ; ++i) {
			fprintf(fout, "%f ", get_v(i, j));
		}
		fprintf(fout, "\n");
	}

	return 0;
}