#ifndef _SEQUENCIAL
#define _SEQUENCIAL

#include "global.h"

#define PAR 0
#define IMPAR 1

// Método de Gauss-Seidel com sobre-relaxação sucessiva
// com iter iterações
void gauss_seidel_seq_w(int iter);

// Método de Gauss-Seidel com sobre-relaxação sucessiva local
// com iter iterações
void gauss_seidel_seq_l(int iter);

#endif