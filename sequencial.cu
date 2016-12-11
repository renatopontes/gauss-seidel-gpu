#include "include/sequencial.h"

// Função auxiliar -----------------------------------------------------------------------

float get_v(int i, int j) {
	if (i < 0) return uo;
	if (i == n1) return ue;
	if (j < 0) return us;
	if (j == n2) return un;

	return malha[i*n2 + j];
}

// Sobre-relaxação sucessiva -------------------------------------------------------------

// Atualiza ponto (i,j) com sobre-relaxação sucessiva. (w fixo)
void atualiza_v_w(int i, int j) {
	float xi = (i+1)*h1;
	float yj = (j+1)*h2;
	float o, e, s, n;

	o = (2.0 + h1 * get_a(xi, yj)) / (4.0 * (1.0 + sq(h1)/sq(h2)));
	e = (2.0 - h1 * get_a(xi, yj)) / (4.0 * (1.0 + sq(h1)/sq(h2)));
	s = (2.0 + h2 * get_b(xi, yj)) / (4.0 * (1.0 + sq(h2)/sq(h1)));
	n = (2.0 - h2 * get_b(xi, yj)) / (4.0 * (1.0 + sq(h2)/sq(h1)));

	malha[i*n2 + j] = (1 - w_fixo) * malha[i*n2 + j] + 
	w_fixo * (o*get_v(i-1,j) + e*get_v(i+1, j) + s*get_v(i, j-1) + n*get_v(i, j+1));
}

// Sobre-relaxação sucessiva local -------------------------------------------------------

// Atualiza ponto (i,j) com sobre-relaxação sucessiva local. (w variável)
void atualiza_v_l(int i, int j) {
	float xi = (i+1)*h1;
	float yj = (j+1)*h2;
	float o, e, s, n, q, w_local;

	o = (2.0 + h1 * get_a(xi, yj)) / (4.0 * (1.0 + sq(h1)/sq(h2)));
	e = (2.0 - h1 * get_a(xi, yj)) / (4.0 * (1.0 + sq(h1)/sq(h2)));
	s = (2.0 + h2 * get_b(xi, yj)) / (4.0 * (1.0 + sq(h2)/sq(h1)));
	n = (2.0 - h2 * get_b(xi, yj)) / (4.0 * (1.0 + sq(h2)/sq(h1)));

	q = 2.0 * (sqrt(e*o) * cos(h1*pi) + sqrt(s*n) * cos(h2 * pi));
	w_local = 2.0 / (1 + sqrt(1 - sq(q)));

	malha[i*n2 + j] = (1 - w_local) * malha[i*n2 + j] + 
	w_local * (o*get_v(i-1,j) + e*get_v(i+1, j) + s*get_v(i, j-1) + n*get_v(i, j+1));
}

// Processa elementos da malha cuja soma dos índices é <paridade>.
// Utiliza a função atualiza_v() para atualizar um elemento.
void processa_malha(int paridade, void (*atualiza_v) (int, int)) {
	for (int i = 0; i < n1; ++i) {
		for (int j = (i+paridade) % 2; j < n2; j += 2) {
			atualiza_v(i, j);
		}
	}
}

// Método de Gauss-Seidel sequencial com relaxação sucessiva iter iterações (w fixo).
// Se modo == LOCAL, é usada relaxação sucessiva local (w variável).
void gauss_seidel_seq(int iter, int modo) {
	void (*atualiza_v) (int, int);
	if (modo == FIXO)
		atualiza_v = atualiza_v_w;
	else
		atualiza_v = atualiza_v_l;

	while(iter--) {
		processa_malha(PAR, atualiza_v);
		processa_malha(IMPAR, atualiza_v);
	}
}