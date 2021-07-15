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
    print("mgrid is", mgrid)
    print("mgrid size is", mgrid.size())

    mgrid = mgrid.reshape(-1, dim)
    print("--------------")
    print("mgrid after resize is", mgrid)
    print("mgrid after resize has shape of", mgrid.size())
    return mgrid

get_mgrid(3, 2)


'''
SAMPLE output is as below.
--------------
single tensor is tensor([-1.,  0.,  1.])
single tensor size is torch.Size([3])
tensor after tuple is (tensor([-1.,  0.,  1.]), tensor([-1.,  0.,  1.]))
--------------
mgrid is tensor([[[-1., -1.],
         [-1.,  0.],
         [-1.,  1.]],

        [[ 0., -1.],
         [ 0.,  0.],
         [ 0.,  1.]],

        [[ 1., -1.],
         [ 1.,  0.],
         [ 1.,  1.]]])
mgrid size is torch.Size([3, 3, 2])
--------------
mgrid after resize is tensor([[-1., -1.],
        [-1.,  0.],
        [-1.,  1.],
        [ 0., -1.],
        [ 0.,  0.],
        [ 0.,  1.],
        [ 1., -1.],
        [ 1.,  0.],
        [ 1.,  1.]])
mgrid after resize has shape of torch.Size([9, 2])

Process finished with exit code 0

'''
