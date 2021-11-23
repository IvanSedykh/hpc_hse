Данные генерируются ноутбуком.

Компиляция:
`module load INTEL/oneAPI_2021_u2_env`

`icc -mkl main.c -o main.exe`



Мейкфайл не работает.

`bash`

`. /opt/intel/oneapi/setvars.sh`

`icc -std=c11 -qmkl main.c -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -o main.exe`

`icc -std=c11 -qmkl main.c -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -o main.exe && ./main.exe`