CC=nvcc
VPATH=include

all: gauss_seidel clean

gauss_seidel: main.obj sequencial.obj paralelo.obj
	mkdir -p out
	$(CC) -o gauss_seidel main.obj sequencial.obj paralelo.obj

main.obj: main.cu sequencial.h paralelo.h global.h
	$(CC) -c main.cu

sequencial.obj: sequencial.cu sequencial.h global.h
	$(CC) -c sequencial.cu

paralelo.obj: paralelo.cu paralelo.h global.h
	$(CC) -c paralelo.cu

clean:
	rm -r *.obj
