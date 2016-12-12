/* Alunos: 																				*/
/*			Renato Pontes Rodrigues														*/
/*			Mateus Ildefonso do Nascimento												*/

/* Para compilar:																		*/
/* 			make [TAM_BLOCO=T]															*/
/* 			T é um inteiro, o tamanho do bloco.	(default: 8)							*/

#include "include/global.h"
#include "include/sequencial.h"
#include "include/paralelo.h"

int n1, n2; // N1 e N2
float *malha;
float un = 5, ue = 10, us = 5, uo = 0; // condições de fronteira.
float h1, h2; // distância entre pontos discretizados, em cada eixo.
const float w_fixo = 1.3; // w para sobre-relaxação não local.
const float pi = 2.0*acos(0); // constante pi.

// Função a, como definida no enunciado do trabalho
__host__ __device__ float get_a(float x, float y) {
	return 500.0 * x * (1.0 - x) * (0.5 - y);
}

// Função b, como definida no enunciado do trabalho
__host__ __device__ float get_b(float x, float y) {
	return 500.0 * y * (1.0 - y) * (x - 0.5);
}

// Inicializa a malha com valores aleatórios
void init_malha() {
	srand(time(NULL));
	for (int i = 0; i < n1*n2; ++i) {
		malha[i] = (1.0 * rand() / RAND_MAX) * 10.0;
	}
}

// Chama a função apropriada e valida os dados de entrada, além de imprimir
// a solução e tempos de execução.
int main(int argc, char **argv) {
	FILE *fout; // arquivo de saída
	TEMPO t; // guardará os tempos de execução do método executado
	int iter; // numero de iterações

	if (argc < 4) {
		SHOW_ERR("Passagem incorreta de parametros.\n\n"
			"\tUso: ./gauss_seidel N1 N2 iter [sw|sl|pw|pl]\n"
			"\tN1: largura da malha\n"
			"\tN2: altura da malha\n"
			"\titer: numero de iteracoes\n"
			"\tsw: processamento sequencial com sobre-relaxacao sucessiva. (default)\n"
			"\tsl: processamento sequencial com sobre-relaxacao sucessiva local.\n"
			"\tpw: processamento paralelo com sobre-relaxacao sucessiva.\n"
			"\tpl: processamento paralelo com sobre-relaxacao sucessiva local.\n");
	}

	// Define os parâmetros globais N1, N2, número de iterações, h1 e h2 da
	// maneira como descritos no enunciado.
	n1 = atoi(argv[1]);
	n2 = atoi(argv[2]);
	iter = atoi(argv[3]);

	h1 = 1.0 / (n1 + 1);
	h2 = 1.0 / (n2 + 1);

	fout = fopen("out/matriz.txt", "w+");
	if (!fout) {
		SHOW_ERR("Nao foi possivel criar arquivo de saida\n");
	}

	// Aloca espaço para a malha no host.
	malha = (float *) malloc(n1 * n2 * sizeof(float));

	init_malha();

	// Chama a função apropriada segundo os argumentos da linha de comando.
	if (argc == 4 || !strcmp(argv[4], "sw")) {
		t = gauss_seidel_seq(iter);
	}
	else if (!strcmp(argv[4], "sl")) {
		t = gauss_seidel_seq(iter, LOCAL);
	}
	else if (!strcmp(argv[4], "pw")) {
		t = gauss_seidel_par(iter);
	}
	else if (!strcmp(argv[4], "pl")) {
		t = gauss_seidel_par(iter, LOCAL);
	}

	// Imprime a malha no mesmo formato mostrado no pdf.
	for(int j = n2; j >= -1; --j) {
		for (int i = -1; i <= n1 ; ++i) {
			fprintf(fout, "%f ", get_v(i, j));
		}
		fprintf(fout, "\n");
	}

	// Imprime os tempos de execução
	printf("%.6f\t%.6f\t%.6f\t%.6f\n", t.ida, t.principal, t.volta, t.total);

	free(malha);

	return 0;
}