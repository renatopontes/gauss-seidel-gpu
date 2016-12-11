CC=nvcc
VPATH=include
FLAGS=-dc

all: gauss_seidel clean

gauss_seidel: main.obj sequencial.obj paralelo.obj
	mkdir -p out
	$(CC) -o gauss_seidel main.obj sequencial.obj paralelo.obj

main.obj: main.cu sequencial.h paralelo.h global.h
	$(CC) -c main.cu $(FLAGS)

sequencial.obj: sequencial.cu sequencial.h global.h
	$(CC) -c sequencial.cu $(FLAGS)

paralelo.obj: paralelo.cu paralelo.h global.h
	$(CC) -c paralelo.cu $(FLAGS)

clean:
	rm -r *.obj
