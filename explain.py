# Purpose: (1) to provide a further explanation of the pixel operation 
           (2) to aid adding new dimensions

import torch

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    print("------------------------------------------Start------------------------------------------")
    print("is", torch.linspace(-1, 1, steps=sidelen))
    print("shape", torch.linspace(-1, 1, steps=sidelen).size())

    # tuple
    print("------------------------------------------tuple-----------------------------------------")
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    print("is", tensors)

    # meshgrid
    print("------------------------------------------mgrid------------------------------------------")
    mgrid_tmp = torch.meshgrid(*tensors)
    print("is", torch.meshgrid(*tensors))

    # stack
    mgrid = torch.stack(mgrid_tmp, dim=-1)
    print("------------------------------------------stack ------------------------------------------")
    print("is", mgrid)
    print("shape", mgrid.size())

    # reshape
    mgrid = mgrid.reshape(-1, dim)
    print("------------------------------------------reshape------------------------------------------")
    print("is", mgrid)
    print("shape", mgrid.size())


    return mgrid

get_mgrid(3, 2)


'''
D:\Anaconda\python.exe D:/Code/SummerResearch/critical_points/tmp_main.py
------------------------------------------Start------------------------------------------
1. When side length is 3, each variable has a tensor of length 3.
is tensor([-1.,  0.,  1.])
shape torch.Size([3])
------------------------------------------tuple-----------------------------------------
2. When we use tuple, we duplicate that tensor
is (tensor([-1.,  0.,  1.]), tensor([-1.,  0.,  1.]))
------------------------------------------mgrid------------------------------------------
3. When we use mesh grid, the first tensor is copied col by col, whereas the second vector is copied row by row.
is (tensor([[-1., -1., -1.],
        [ 0.,  0.,  0.],
        [ 1.,  1.,  1.]]), tensor([[-1.,  0.,  1.],
        [-1.,  0.,  1.],
        [-1.,  0.,  1.]]))
------------------------------------------stack ------------------------------------------
4. When we use stack, two tensors were combined as one. The mapping is (col, row, tensor) -> (row, matrix, col).
           For example, the 1st col in the 1st row in the 1st tensor maps to the 1st row in the 1st matrix with the 1st col.
           
is tensor([[[-1., -1.],
         [-1.,  0.],
         [-1.,  1.]],

        [[ 0., -1.],
         [ 0.,  0.],
         [ 0.,  1.]],

        [[ 1., -1.],
         [ 1.,  0.],
         [ 1.,  1.]]])
shape torch.Size([3, 3, 2]) # 3 tensors of size 3 (row) by 2 (col)
------------------------------------------reshape------------------------------------------
is tensor([[-1., -1.],
        [-1.,  0.],
        [-1.,  1.],
        [ 0., -1.],
        [ 0.,  0.],
        [ 0.,  1.],
        [ 1., -1.],
        [ 1.,  0.],
        [ 1.,  1.]])
shape torch.Size([9, 2]) # 1 tensor of size 9 (row) by 2 (col)

Process finished with exit code 0
'''



