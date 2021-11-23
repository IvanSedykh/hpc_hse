CC = icc
EXECS=main.exe 
CFLAGS=-mkl -qopenmp -O3
LFLAGS=

release: $(EXECS)

run: release
	@for i in $(EXECS); do ./$$i; done

main.exe: main.c
	$(CC) $(CFLAGS) $^ $(LFLAGS) -o $@

clean:
	rm -f core.* *.o *.exe
