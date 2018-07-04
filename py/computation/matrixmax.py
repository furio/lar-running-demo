import numpy as np
from pycuda import driver, compiler, gpuarray, tools

kernel_code_template = """
__global__ void FindMaxRow(float *outmax, float *indices, float *indptr, float *nnz)
{
    int tx = threadIdx.x;

    // Pvalue is used to store the element of the matrix
    // that is computed by the thread
    float Pvalue = 0;

    // Each thread loads one row of M and one column of N, 
    //   to produce one element of P.
    for (int k = 0; k < %(MATRIX_SIZE)s; ++k) {
        float Aelement = a[ty * %(MATRIX_SIZE)s + k];
        float Belement = b[k * %(MATRIX_SIZE)s + tx];
        Pvalue += Aelement * Belement;
    }

    // Write the matrix to device memory;
    // each thread writes one element
    c[ty * %(MATRIX_SIZE)s + tx] = Pvalue;
}
"""

def findmaxrows(indices, indptr, nnz):
    # -- initialize the device
    import pycuda.autoinit