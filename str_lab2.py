# -*- coding: utf-8 -*-

import numpy as np
import time
from scipy.linalg.blas import cgemm

# Функция для генерации матрицы комплексных чисел
def generate_matrix(size):
    return np.random.rand(size, size) + 1j * np.random.rand(size, size).astype(np.complex64)

# Размер матрицы
N = 1024

# Генерация двух матриц
A = generate_matrix(N)
B = generate_matrix(N)

def multiply_matrices_basic(A, B):
    C = np.zeros((N, N), dtype=np.complex64)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]
    return C

# Замер времени выполнения
start_time = time.time()
C_basic = multiply_matrices_basic(A, B)
elapsed_time_basic = time.time() - start_time

start_time = time.time()
C_blas = cgemm(1.0, A, B)
elapsed_time_blas = time.time() - start_time

start_time = time.time()
C_numpy = np.dot(A, B)
elapsed_time_numpy = time.time() - start_time

complexity = 2 * (N ** 3)

def calculate_performance(time_elapsed):
    return complexity / (time_elapsed * 1e6)

performance_basic = calculate_performance(elapsed_time_basic)
performance_blas = calculate_performance(elapsed_time_blas)
performance_numpy = calculate_performance(elapsed_time_numpy)

print("Сизов Константин Васильевич, РПИб-о23")

print("Время выполнения формулы линейной алгебры:", elapsed_time_basic, "секунды")
print("Производительность формулы линейной алгебры:", performance_basic, "MFlops")
print("Время выполнения BLAS:", elapsed_time_blas, "секунды")
print("Производительность BLAS:", performance_blas, "MFlops")
print("Время выполнения NumPy:", elapsed_time_numpy, "секунды")
print("Производительность NumPy:", performance_numpy, "MFlops")