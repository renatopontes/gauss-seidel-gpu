CC=nvcc
VPATH=include
TAM_BLOCO=8
FLAGS=-dc -DTAM_BLOCO=$(TAM_BLOCO)

all: gauss_seidel

gauss_seidel: main.obj sequencial.obj paralelo.obj
	$(CC) -o gauss_seidel main.obj sequencial.obj paralelo.obj
	mkdir -p out
	mkdir -p obj
	mv *.obj obj/

main.obj: main.cu sequencial.h paralelo.h global.h
	$(CC) -c main.cu $(FLAGS)

sequencial.obj: sequencial.cu sequencial.h global.h
	$(CC) -c sequencial.cu $(FLAGS)

paralelo.obj: paralelo.cu paralelo.h global.h
	$(CC) -c paralelo.cu $(FLAGS)

clean:
	rm -rf *.lib *.exp *.exe *.obj obj

clean_output:
	rm -rf out/*