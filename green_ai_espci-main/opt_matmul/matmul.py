
import numpy as np
#from numba import jit

#@jit(nopython=True, cache=True, fastmath=False)
def matmul_naive(A, B, M, N, K):
	C = np.zeros((M,N))
	for i in range(0,M):
		for j in range(0,N):
			for k in range(0,K):
				C[i,j] += A[i,k] * B[k,j]
	return C
	
#@jit(nopython=True, cache=True, fastmath=False)
def matmul_numpy_sum(A, B, M, N, K):
	C = np.zeros((M,N))
	for i in range(0,M):
		for j in range(0,N):
			C[i,j] = np.sum(A[i,:]*B[:,j])
				
	return C


M = 512
N = 512
K = 512

np.random.seed(0)

A = (np.random.random((M,K))-0.5)*0.1
B = (np.random.random((K,N))-0.5)-0.1

#### Select one ####
C = matmul_naive(A,B,M,N,K)
#C = matmul_numpy_sum(A, B, M, N, K)
#C = A@B
#C = np.matmul(A,B)
#C = np.dot(A,B)

print (C[M//2,N//2])









