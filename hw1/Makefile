# НЕ РАБОТАЕТ


CC = icc
EXECS=main.exe 
CFLAGS=-std=c11 -qmkl -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
LFLAGS=-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk

release: $(EXECS)

run: release
	. /opt/intel/oneapi/setvars.sh
	@for i in $(EXECS); do ./$$i; done

main.exe: main.c
	. /opt/intel/oneapi/setvars.sh
	$(CC) -std=c11 -qmkl $^ -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -o $@

clean:
	/bin/rm -f core.* *.o *.exe

# main.exe: main.o
# 	$(CC) -O2 $^ $(LFLAGS) -o $@

# %.o: %.c
# 	$(CC) $(CFLAGS) -O2 -c -o $@  $<