import torch
import torch.distributions as dist

class Normal(dist.Normal):
    
    def __init__(self, alpha, loc, scale):
        
        if scale > 20.:
            self.optim_scale = scale.clone().detach().requires_grad_()
        else:
            self.optim_scale = torch.log(torch.exp(scale) - 1).clone().detach().requires_grad_()
        
        
        super().__init__(loc, torch.nn.functional.softplus(self.optim_scale))
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.loc, self.optim_scale]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        ps = [p.clone().detach().requires_grad_() for p in self.Parameters()]
         
        return Normal(*ps)
    
    def log_prob(self, x):
        
        self.scale = torch.nn.functional.softplus(self.optim_scale)
        
        return super().log_prob(x)
        
def push_addr(alpha, value):
    return alpha + value

import torch

def add(*args):
    return args[0] + args[1]

def subtract(*args):
    return args[0] - args[1]

def multiply(*args):
    return args[0] * args[1]

def divide(*args):
    return args[0] / args[1]

def vector(*args):
    all_are_numbers = True
    for item in args:
        if not torch.is_tensor(item):
            all_are_numbers = False
            break
    if all_are_numbers:
        try:
            if (len(args[0].size()) > 0) and (args[0].size(0) > 1):
                tensor_to_return = torch.stack(args)
            else:
                tensor_to_return = torch.Tensor(args)
            return tensor_to_return
        except Exception:
            return args
    else:
        return args

def get(*args):
    if type(args[0]) is torch.Tensor:
        return args[0][args[1].long().item()]
    if type(args[0]) is list:
        return args[0][args[1].long().item()]
    if type(args[1]) is torch.Tensor:
        index = args[1].float().item()
    else:
        index = args[1]
    return args[0][index]

def conj(*args):
    if type(args[0]) is list:
        return args[0] + [args[1]]
    else:
        if len(args[0]) == 0:
            return args[1]
        to_return = args[0]
        if args[1] is not list:
            if type(args[1]) == torch.Tensor and len(args[1].size()) == 0:
                to_return = append(args[1].unsqueeze(0), to_return)
            else:
                to_return = append(args[1], to_return)
        else:
            for item in args[1]:
                if type(item) == torch.Tensor and len(item.size()) == 0:
                    item = item.unsqueeze(0)
                to_return = append(item, to_return)
    return to_return

def put(*args):
    if type(args[0]) == dict:
        new_dict = args[0].copy()
        if type(args[1]) is torch.Tensor:
            new_dict[args[1].float().item()] = args[2]
        else:
            new_dict[args[1]] = args[2]
        return new_dict
    elif type(args[0]) == torch.Tensor:
        new_list = args[0].clone()
        new_list[args[1].long()] = args[2]
        return new_list

    else:
        new_list = args[0].copy()
        new_list[args[1].long()] = args[2]
        return new_list

def first(*args):
    return args[0][0]

def second(*args):
    return args[0][1]

def rest(*args):
    return args[0][1:]

def last(*args):
    return args[0][-1]

def append(*args):
    print('append input check', args)
    if type(args[0]) == list:
        return args[0].append(args[1])
    else:
        if len(args[0].size()) > len(args[1].size()):
            return torch.cat((args[0], args[1].unsqueeze(0)), dim=0)
        else:
            return torch.cat((args[0], args[1]), dim=0)

def hashmap(*args):
    hashmap_to_return = {}
    for index in range(int(len(args) / 2)):
        try:
            hashmap_to_return[float(args[2*index])] = args[2*index+1]
        except Exception:
            hashmap_to_return[args[2*index]] = args[2*index+1]
    return hashmap_to_return

def less(*args):
    return args[0] < args[1]

def greater(*args):
    return args[0] > args[1]

def equal(*args):
    return args[0] == args[1]

def sqrt(*args):
    return torch.sqrt(args[0])

def normal(*args):
    return torch.distributions.Normal(args[0], args[1])

def beta(*args):
    return torch.distributions.Beta(args[0], args[1])

def exponential(*args):
    return torch.distributions.Exponential(args[0])

def uniform(*args):
    return torch.distributions.Uniform(args[0], args[1])

def discrete(*args):
    return torch.distributions.Categorical(probs=torch.Tensor(args[0]))

def flip(*args):
    return torch.distributions.Categorical(probs=torch.Tensor([1-args[0], args[0]]))

def sample(*args):
    return args[0].sample()

def mattranspose(*args):
    if len(args[0].size()) > 1:
        return args[0].transpose(0,1)
    else:
        return args[0].unsqueeze(1).transpose(0,1)

def mattanh(*args):
    return torch.tanh(args[0])

def matadd(*args):
    return args[0] + args[1]

def matmul(*args):
    if len(args[0].size()) == 1:
        return torch.matmul(args[0].unsqueeze(1), args[1])
    return torch.matmul(args[0], args[1])

def matrepmat(*args):
    if len(args[0].size()) > 1:
        repeated = args[0].repeat(args[2].int().item(), args[1].int().item())
    else:
        repeated = args[0].unsqueeze(1).repeat(args[1].int().item(), args[2].int().item())
    return repeated

def iffunction(*args):
    if args[0]:
        return args[1]
    else:
        return args[2]

def empty(*args):
    return len(args[0]) == 0

def log(*args):
    return torch.log(args[0])

env = {
           'normal' : Normal,
           'push-address' : push_addr,
           '+': add,
            '-': subtract,
            '*': multiply,
            '/': divide,
            '<': less,
            '>': greater,
            '=': equal,
            'sqrt': sqrt,
            'vector': vector,
            'get': get,
            'put': put,
            'first': first,
            'rest': rest,
            'last': last,
            'append': append,
            'hash-map': hashmap,
            'normal': normal,
            'sample': sample,
            'beta': beta,
            'observe': sample,
            'exponential': exponential,
            'uniform-continuous': uniform,
            'second': second,
            'discrete': discrete,
            'mat-transpose': mattranspose,
            'mat-add': matadd,
            'mat-mul': matmul,
            'mat-repmat': matrepmat,
            'mat-tanh': mattanh,
            'empty?': empty,
            'conj': conj,
            'flip': flip,
            'log': log,
            'peek': last,
       }






