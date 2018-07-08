# -*- coding: utf-8 -*-
"""
The MIT License
===============
    
Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
'Software'), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    
"""

import pyculib as p
from scipy.sparse import csr_matrix
import numpy as np

def matrixProduct(CSRm1,CSRm2):
    CSRm = CSRm1 * CSRm2
    return CSRm

def matrixProductAccel(matrixOne, matrixTwo):
    sparseIstance = p.sparse.Sparse()
    
    cudaCSR1 = p.sparse.csr_matrix((matrixOne.data, matrixOne.indices, matrixOne.indptr), shape=matrixOne.shape)
    cudaCSR2 = p.sparse.csr_matrix((matrixTwo.data, matrixTwo.indices, matrixTwo.indptr), shape=matrixTwo.shape)
    
    cudaCSRm = sparseIstance.csrgemm_ez(cudaCSR1, cudaCSR2)
    return csr_matrix((cudaCSRm.data, cudaCSRm.indices, cudaCSRm.indptr), shape=cudaCSRm.shape, dtype=np.int32)

def csrTranspose(CSRm):
    CSRm = CSRm.T
    return CSRm
