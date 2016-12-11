#include "include/sequencial.h"

// Sobre-relaxação sucessiva -------------------------------------------------------------

// Atualiza ponto (i,j) com sobre-relaxação sucessiva. (w fixo)
void atualiza_v_w(int i, int j) {
	COORD_MALHA pos = valor(i, j);
	float o, e, s, n;

	o = (2.0 + h1 * get_a(pos.xi, pos.yj)) / (4.0 * (1.0 + sq(h1)/sq(h2)));
	e = (2.0 - h1 * get_a(pos.xi, pos.yj)) / (4.0 * (1.0 + sq(h1)/sq(h2)));
	s = (2.0 + h2 * get_b(pos.xi, pos.yj)) / (4.0 * (1.0 + sq(h2)/sq(h1)));
	n = (2.0 - h2 * get_b(pos.xi, pos.yj)) / (4.0 * (1.0 + sq(h2)/sq(h1)));

	malha[i*n2 + j] = (1 - w_fixo) * malha[i*n2 + j] + 
	w_fixo * (o*get_v(i-1,j) + e*get_v(i+1, j) + s*get_v(i, j-1) + n*get_v(i, j+1));
}

// Processa elementos da malha cuja soma dos índices é <paridade>.
// Utiliza sobre-relaxação sucessiva.
void processa_malha_w(int paridade) {
	for (int i = 0; i < n1; ++i) {
		for (int j = (i+paridade) % 2; j < n2; j += 2) {
			atualiza_v_w(i, j);
		}
	}
}

void gauss_seidel_seq_w(int iter) {
	while(iter--) {
		processa_malha_w(PAR);
		processa_malha_w(IMPAR);
	}
}

// Sobre-relaxação sucessiva local -------------------------------------------------------

// Atualiza ponto (i,j) com sobre-relaxação sucessiva local. (w variável)
void atualiza_v_l(int i, int j) {
	COORD_MALHA pos = valor(i, j);
	float o, e, s, n, q, w_local;

	o = (2.0 + h1 * get_a(pos.xi, pos.yj)) / (4.0 * (1.0 + sq(h1)/sq(h2)));
	e = (2.0 - h1 * get_a(pos.xi, pos.yj)) / (4.0 * (1.0 + sq(h1)/sq(h2)));
	s = (2.0 + h2 * get_b(pos.xi, pos.yj)) / (4.0 * (1.0 + sq(h2)/sq(h1)));
	n = (2.0 - h2 * get_b(pos.xi, pos.yj)) / (4.0 * (1.0 + sq(h2)/sq(h1)));

	q = 2.0 * (sqrt(e*o) * cos(h1*PI) + sqrt(s*n) * cos(h2 * PI));
	w_local = 2.0 / (1 + sqrt(1 - sq(q)));

	malha[i*n2 + j] = (1 - w_local) * malha[i*n2 + j] + 
	w_local * (o*get_v(i-1,j) + e*get_v(i+1, j) + s*get_v(i, j-1) + n*get_v(i, j+1));
}

// Processa elementos da malha cuja soma dos índices é <paridade>.
// Utiliza sobre-relaxação sucessiva local.
void processa_malha_l(int paridade) {
	for (int i = 0; i < n1; ++i) {
		for (int j = (i+paridade) % 2; j < n2; j += 2) {
			atualiza_v_l(i, j);
		}
	}
}

// Método de Gauss-Seidel com sobre-relaxação sucessiva local
void gauss_seidel_seq_l(int iter) {
	while(iter--) {
		processa_malha_l(PAR);
		processa_malha_l(IMPAR);
	}
}