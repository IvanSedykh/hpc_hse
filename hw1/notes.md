`bash`

`. /opt/intel/oneapi/setvars.sh`

`icc -std=c11 -qmkl main.c -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -o main.exe`

`cc -std=c11 -qmkl main.c -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -o main.exe && ./main.exe`