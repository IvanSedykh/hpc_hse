#!/bin/bash
#SBATCH --job-name=hw2            # Название задачи
#SBATCH --error=logs/hw2-%j.err        # Файл для вывода ошибок
#SBATCH --output=logs/hw2-%j.log       # Файл для вывода результатов
#SBATCH --time=00:10:00            # Максимальное время выполнения
#SBATCH --ntasks=1                # Количество MPI процессов
#SBATCH --nodes=1                  # Требуемое кол-во узлов
#SBATCH --gpus=0                   # Требуемое кол-во GPU

module load INTEL/oneAPI_2021_u2_env                # Загрузка модуля

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun ./main.exe             # Выполнение расчёта
python3 ./validate.py