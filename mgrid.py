import torch

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    print("--------------")
    print("single tensor is", torch.linspace(-1, 1, steps=sidelen))
    print("single tensor size is", torch.linspace(-1, 1, steps=sidelen).size())

    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    print("tensor after tuple is", tensors)

    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    print("--------------")
    print("mgrid before stack is", torch.meshgrid(*tensors))

    print("--------------")
    print("mgrid is", mgrid)
    print("mgrid size is", mgrid.size())

    mgrid = mgrid.reshape(-1, dim)
    print("--------------")
    print("mgrid after resize is", mgrid)
    print("mgrid after resize has shape of", mgrid.size())
    return mgrid

get_mgrid(2, 2)

'''
D:\Anaconda\python.exe D:/Code/SummerResearch/critical_points/main.py
--------------
single tensor is tensor([-1.,  1.])
single tensor size is torch.Size([2])
tensor after tuple is (tensor([-1.,  1.]), tensor([-1.,  1.]))
--------------
mgrid before stack is (tensor([[-1., -1.],
        [ 1.,  1.]]), tensor([[-1.,  1.],
        [-1.,  1.]]))
--------------
mgrid is tensor([[[-1., -1.],
         [-1.,  1.]],

        [[ 1., -1.],
         [ 1.,  1.]]])
mgrid size is torch.Size([2, 2, 2])
--------------
mgrid after resize is tensor([[-1., -1.],
        [-1.,  1.],
        [ 1., -1.],
        [ 1.,  1.]])
mgrid after resize has shape of torch.Size([4, 2])

Process finished with exit code 0
'''
