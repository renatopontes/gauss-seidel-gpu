/* Alunos: 																				*/
/*			Renato Pontes Rodrigues														*/
/*			Mateus Ildefonso do Nascimento												*/

#ifndef _SEQUENCIAL
#define _SEQUENCIAL

#include "global.h"

// Método de Gauss-Seidel sequencial com sobre-relaxação sucessiva
// com iter iterações. Utiliza w variável se modo == LOCAL.
// Retorna uma estrutura TEMPO com os tempos de execução.
TEMPO gauss_seidel_seq(int iter, int modo=FIXO);

// Retorna o valor da malha na posição (i, j).
float get_v(int i, int j);

#endif