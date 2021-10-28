#!/bin/bash
#SBATCH --job-name=hw1            # Название задачи
#SBATCH --error=hw1-%j.err        # Файл для вывода ошибок
#SBATCH --output=hw1-%j.log       # Файл для вывода результатов
#SBATCH --time=12:00:00            # Максимальное время выполнения
#SBATCH --ntasks=1                # Количество MPI процессов
#SBATCH --nodes=1                  # Требуемое кол-во узлов
#SBATCH --gpus=0                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=1          # Требуемое кол-во CPU

module load INTEL/oneAPI_2021_u2_env                # Загрузка модуля
srun ./main.exe             # Выполнение расчёта