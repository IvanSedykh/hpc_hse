#!/bin/bash
#SBATCH --job-name=hw3            # Название задачи
#SBATCH --error=logs/hw3-%j.err        # Файл для вывода ошибок
#SBATCH --output=logs/hw3-%j.log       # Файл для вывода результатов
#SBATCH --time=00:10:00            # Максимальное время выполнения
#SBATCH --ntasks-per-node=1                # Количество MPI процессов
#SBATCH --nodes=1                  # Требуемое кол-во узлов
#SBATCH --gpus=0                   # Требуемое кол-во GPU

module load INTEL/oneAPI_2021_u2_env                # Загрузка модуля

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export I_MPI_PMI_LIBRARY=/lib64/libpmi.so  # какое-то колдунство
srun ./main.exe             # Выполнение расчёта
python3 ./validate.py
srun hostname