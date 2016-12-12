#!/usr/bin/env python3

# Alunos:
#			Renato Pontes Rodrigues
#			Mateus Ildefonso do Nascimento
#
# Este script roda a aplicação várias vezes com diferentes combinações de parâmetros
# de entrada, e formata a saída num arquivo CSV.

import os
import subprocess as sp

tam_blocos = [8, 16]
dims = [(100,100), (300,300), (500,500), (1000,1000)]
reps = [100, 500, 1000, 5000]
modos = [('sw', 'pw'), ('sl', 'pl')]

n_tests = len(tam_blocos)*len(dims)*len(reps)*len(modos)
i = 0;

def update_status(msg):
	print(('\r{}/{}   ({})' + ' '*13).format(i, n_tests, msg), end='')

os.chdir('..')
os.system('make clean -s')
os.system('make clean_output -s')

stats = open('stats.csv', 'w+', buffering=1)

exe = 'gauss_seidel'
if (os.name == 'posix'):
	exe = './' + exe

header = ['reps', 'N1xN2', 'modo', 'total seq', 'tempo ida', 'tempo kernel', 'tempo volta', 'total par', 'acel']

print('\r{}/{}'.format(i, n_tests), end='')
for tam_bloco in tam_blocos:
	update_status("Compilando...")
	os.system('make -s TAM_BLOCO={} >nul 2>&1'.format(tam_bloco))
	stats.write('{0}x{0}\n'.format(tam_bloco))
	stats.write(('{:<12};'*len(header) + '\n').format(*header))
	for rep in reps:
		for dim in dims:
			for seq, par in modos:
				cmd = '{} {} {} {} {}'.format(exe, dim[0], dim[1], rep, seq)
				update_status("Rodando sequencial...")
				out_seq = sp.getoutput(cmd)
				update_status("Rodando sequencial... Pronto")
				try:
					t_seq = float(out_seq.strip().split()[-1])
				except ValueError:
					stats.write(out_par + '\n')
				
				os.system('make clean_output -s')

				cmd = '{} {} {} {} {}'.format(exe, dim[0], dim[1], rep, par)
				update_status("Rodando paralelo...")
				out_par = sp.getoutput(cmd)
				update_status("Rodando paralelo... Pronto")
				try:
					t_ida,t_kernel,t_volta,t_par = map(float, out_par.strip().split())
				except ValueError:
					err = 'ERRO'
					stats.write(('{:<12};'*(len(header)-1) + '{:<12.6f}\n').format(
					rep, '{}x{}'.format(dim[0], dim[1]), seq[-1], t_seq, err, err,
					err, err, 0))
				else:
					stats.write(('{:<12};'*(len(header)-1) + '{:<12.6f}\n').format(
					rep, '{}x{}'.format(dim[0], dim[1]), seq[-1], t_seq, t_ida, t_kernel,
					t_volta, t_par, t_seq/t_par))

				os.system('make clean_output -s')


				i+=1
				print('\r{}/{}'.format(i, n_tests), end='')
			#
			stats.write('\n')
		#
	#
	stats.write('\n');

