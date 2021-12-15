CC = mpiicc
EXECS=main.exe 
CFLAGS=-mkl -qopenmp
LFLAGS=

release: $(EXECS)

run: release
	@for i in $(EXECS); do mpirun -n 2 -genv OMP_NUM_THREADS=1 -genv I_MPI_PIN_DOMAIN=omp ./$$i; done

main.exe: main.c
	$(CC) $(CFLAGS) $^ $(LFLAGS) -o $@

clean:
	rm -f core.* *.o *.exe
