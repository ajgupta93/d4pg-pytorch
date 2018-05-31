import torch
from torch.autograd import Variable

def to_numpy(var):
    return var.data.numpy()#var.cpu().data.numpy() if use_cuda else var.data.numpy()

def to_tensor(x, volatile=False, requires_grad=True, dtype=torch.FloatTensor):
    x = torch.from_numpy(x).float()
    x = Variable(x, requires_grad=requires_grad).type(dtype)
    return x