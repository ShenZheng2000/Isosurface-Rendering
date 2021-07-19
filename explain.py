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
is tensor([-1.,  0.,  1.])
shape torch.Size([3])
------------------------------------------tuple-----------------------------------------
is (tensor([-1.,  0.,  1.]), tensor([-1.,  0.,  1.]))
------------------------------------------mgrid------------------------------------------
is (tensor([[-1., -1., -1.],
        [ 0.,  0.,  0.],
        [ 1.,  1.,  1.]]), tensor([[-1.,  0.,  1.],
        [-1.,  0.,  1.],
        [-1.,  0.,  1.]]))
------------------------------------------stack ------------------------------------------
is tensor([[[-1., -1.],
         [-1.,  0.],
         [-1.,  1.]],

        [[ 0., -1.],
         [ 0.,  0.],
         [ 0.,  1.]],

        [[ 1., -1.],
         [ 1.,  0.],
         [ 1.,  1.]]])
shape torch.Size([3, 3, 2])
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
shape torch.Size([9, 2])

Process finished with exit code 0
'''



