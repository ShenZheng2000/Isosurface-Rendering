import torch


def create_matrix_noreshape(side, n): # n are numbers of variables

    # Create a 4d tensor of side * side * 1 * n
    my_tensor = torch.zeros((side, side, 1, n))

    # fill 2d tensors with value
    for i in range(0, side):
        for j in range(0, side):
            my_tensor[i][j][0][0] = 1 # the 1st variable
            my_tensor[i][j][0][1] = 1 # the 2nd variable
            my_tensor[i][j][0][2] = 1 # the 3rd variable
            my_tensor[i][j][0][3] = 1 # the 4th variable
            my_tensor[i][j][0][4] = 1 # the 5th variable

    # reshape tensor to 2d
    #my_tensor = my_tensor.reshape(-1,n)

    # return
    return my_tensor.size()

def create_matrix(side, n): # n are numbers of variables

    # Create a 4d tensor of side * side * 1 * n
    my_tensor = torch.zeros((side, side, 1, n))

    # fill 2d tensors with value
    for i in range(0, side):
        for j in range(0, side):
            my_tensor[i][j][0][0] = 1 # the 1st variable
            my_tensor[i][j][0][1] = 1 # the 2nd variable
            my_tensor[i][j][0][2] = 1 # the 3rd variable
            my_tensor[i][j][0][3] = 1 # the 4th variable
            my_tensor[i][j][0][4] = 1 # the 5th variable

    # reshape tensor to 2d
    my_tensor = my_tensor.reshape(-1,n)

    # return
    return my_tensor.size()

print(create_matrix_noreshape(side = 256, n = 5))
print(create_matrix(side = 256, n = 5))



