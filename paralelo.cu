/* Alunos: 																				*/
/*			Renato Pontes Rodrigues														*/
/*			Mateus Ildefonso do Nascimento												*/

#include "include/paralelo.h"

// Funções auxiliares --------------------------------------------------------------------

// Verifica se a posição (i,j) da malha é válida.
__device__ int pos_valida(int i, int j, GLOBALS *g) {
	return (i >= 0) && (i < g->n1) && (j >= 0) && (j < g->n2);
}

// Retorna a solução aproximada do ponto (i,j) da malha.
__device__ float get_v(float *malha, int i, int j, GLOBALS *g) {
	if (i < 0) return g->uo;
	if (i == g->n1) return g->ue;
	if (j < 0) return g->us;
	if (j == g->n2) return g->un;

	return malha[i*g->n2 + j];
}

// Copia os parâmetros globais do problema para uma estrutura.
void collect_globals(GLOBALS *g) {
	g->n1 = n1; g->n2 = n2;
	g->h1 = h1; g->h2 = h2;
	g->un = un; g->ue = ue; g->us = us; g->uo = uo;
	g->w_fixo = w_fixo;
	g->pi = pi;
}

// Sobre-relaxação sucessiva -------------------------------------------------------------

// Cada thread atualiza uma posição da malha de acordo com o método
// de Gauss-Seidel com sobre-relaxação sucessiva descrito no enunciado.
__global__ void processa_malha_w(float *malha, const int paridade, GLOBALS *g) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (pos_valida(i, j, g) && ((i+j) % 2 == paridade)) {
    	int n2 = g->n2;
		float o, e, s, n;
		float h1 = g->h1, h2 = g->h2;
		float w_fixo = g->w_fixo;
		float xi = (i+1.0)*g->h1;
		float yj = (j+1.0)*g->h2;

		o = (2.0 + h1 * get_a(xi, yj)) / (4.0 * (1.0 + sq(h1)/sq(h2)));
		e = (2.0 - h1 * get_a(xi, yj)) / (4.0 * (1.0 + sq(h1)/sq(h2)));
		s = (2.0 + h2 * get_b(xi, yj)) / (4.0 * (1.0 + sq(h2)/sq(h1)));
		n = (2.0 - h2 * get_b(xi, yj)) / (4.0 * (1.0 + sq(h2)/sq(h1)));

		malha[i*n2 + j] = (1 - w_fixo) * malha[i*n2 + j] + w_fixo *
			(o*get_v(malha,i-1,j,g) + e*get_v(malha,i+1,j,g) +
			s*get_v(malha,i,j-1,g) + n*get_v(malha,i,j+1,g));
    }
}

// Sobre-relaxação sucessiva local -------------------------------------------------------

// Cada thread atualiza uma posição da malha de acordo com o método
// de Gauss-Seidel com sobre-relaxação sucessiva local descrito no enunciado.
__global__ void processa_malha_l(float *malha, const int paridade, GLOBALS *g) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (pos_valida(i, j, g) && ((i+j) % 2 == paridade)) {
    	int n2 = g->n2;
		float o, e, s, n, q, w_local;
		float h1 = g->h1, h2 = g->h2;
		float xi = (i+1.0)*g->h1;
		float yj = (j+1.0)*g->h2;
		float pi = g->pi;

		o = (2.0 + h1 * get_a(xi, yj)) / (4.0 * (1.0 + sq(h1)/sq(h2)));
		e = (2.0 - h1 * get_a(xi, yj)) / (4.0 * (1.0 + sq(h1)/sq(h2)));
		s = (2.0 + h2 * get_b(xi, yj)) / (4.0 * (1.0 + sq(h2)/sq(h1)));
		n = (2.0 - h2 * get_b(xi, yj)) / (4.0 * (1.0 + sq(h2)/sq(h1)));

		q = 2.0 * (sqrt(e*o) * cos(h1*pi) + sqrt(s*n) * cos(h2 * pi));
		w_local = 2.0 / (1 + sqrt(1 - sq(q)));

		malha[i*n2 + j] = (1 - w_local) * malha[i*n2 + j] + w_local *
			(o*get_v(malha,i-1,j,g) + e*get_v(malha,i+1,j,g) +
			s*get_v(malha,i,j-1,g) + n*get_v(malha,i,j+1,g));
    }
}

// Código para alocar memória no device e rodar os kernels -------------------------------

// Lança o kernel e mede os tempos de execução.
// Retorna uma estrutura TEMPO com os tempos de execução.
TEMPO gauss_seidel_par(int iter, int modo) {
	TEMPO t;
	float inicio, fim, tempo_kernel;
	cudaEvent_t start, stop;
	GLOBALS *gd, gh;
	float *malha_dev;
	int n_bytes = n1 * n2 * sizeof(float);

	GET_TIME(inicio);
	collect_globals(&gh); // faz cópia dos parametros globais

	// Aloca espaço para a malha na memória do device
	CUDA_SAFE_CALL(cudaMalloc((void**) &malha_dev, n_bytes));
	CUDA_SAFE_CALL(cudaMemcpy(malha_dev, malha, n_bytes, cudaMemcpyHostToDevice));

	// Aloca as variáveis globais na memória do device
	CUDA_SAFE_CALL(cudaMalloc((void**) &gd, sizeof(GLOBALS)));
	CUDA_SAFE_CALL(cudaMemcpy(gd, &gh, sizeof(GLOBALS), cudaMemcpyHostToDevice));
	
	dim3 n_threads(TAM_BLOCO, TAM_BLOCO);
	int a, b;
	a = n1/n_threads.x;
	a += n1 % n_threads.x ? 1 : 0;
	b = n2/n_threads.y;
	b += n2 % n_threads.y ? 1 : 0;
    dim3 blocos_grade(a, b);

    GET_TIME(fim);

    t.set_ida(fim-inicio);

    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));

    if (modo == FIXO) {
    	CUDA_SAFE_CALL(cudaEventRecord(start));
		while(iter--) {
			processa_malha_w<<<blocos_grade, n_threads>>>(malha_dev, PAR, gd);
			CUDA_SAFE_CALL(cudaGetLastError());
			processa_malha_w<<<blocos_grade, n_threads>>>(malha_dev, IMPAR, gd);
			CUDA_SAFE_CALL(cudaGetLastError());
		}
	} else {
		CUDA_SAFE_CALL(cudaEventRecord(start));
		while(iter--) {
			processa_malha_l<<<blocos_grade, n_threads>>>(malha_dev, PAR, gd);
			CUDA_SAFE_CALL(cudaGetLastError());
			processa_malha_l<<<blocos_grade, n_threads>>>(malha_dev, IMPAR, gd);
			CUDA_SAFE_CALL(cudaGetLastError());
		}
	}

	CUDA_SAFE_CALL(cudaEventRecord(stop));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&tempo_kernel, start, stop)); // (tempo em ms)
    t.set_principal(tempo_kernel/1000);

	GET_TIME(inicio);
	CUDA_SAFE_CALL(cudaMemcpy(malha, malha_dev, n_bytes, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(malha_dev));
	CUDA_SAFE_CALL(cudaFree(gd));
	CUDA_SAFE_CALL(cudaDeviceReset());
	GET_TIME(fim);

	t.set_volta(fim-inicio);

	return t;
}