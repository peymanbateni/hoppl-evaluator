from primitives import env as penv
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from pyrsistent import pmap,plist
import math
import operator as op
import torch
import pickle

Symbol = str              # A Scheme Symbol is implemented as a Python str
Number = (int, float)     # A Scheme Number is implemented as a Python int or float
Atom   = (Symbol, Number) # A Scheme Atom is a Symbol or Number
List   = list             # A Scheme List is implemented as a Python list
Exp    = (Atom, List)     # A Scheme expression is an Atom or List

class Env(dict):
    "An environment: a dict of {'var': val} pairs, with an outer Env."
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer
    def find(self, var):
        "Find the innermost Env where var appears."
        return self if (var in self) else self.outer.find(var)

class Procedure(object):
    "A user-defined Scheme procedure."
    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env
    def __call__(self, *args):
        env_to_pass = pmap(self.env)
        env_to_pass = env_to_pass.update(dict(zip(self.parms[1:], args)))
        return evaluate(self.body, env_to_pass)
    def call(self, *args):
        env_to_pass = pmap(self.env)
        env_to_pass = env_to_pass.update(dict(zip(self.parms[1:], args)))
        return evaluate(self.body, env_to_pass)

def standard_env():
    "An environment with some Scheme standard procedures."
    env = pmap(penv)
    env = env.update({'alpha' : ''}) 
    #env = env.update(vars(math)) # sin, cos, sqrt, pi, ...
    #env = env.update({
    #    'sqrt': torch.sqrt,
    #    '+': op.add, '-':op.sub, '*':op.mul, '/':op.truediv, 
    #    '>': op.gt, '<':op.lt, '>=':op.ge, '<=':op.le, '=':op.eq, 
    #    'abs':     abs,
    #    'append':  op.add,  
    #    'apply':   lambda proc, args: proc(*args),
    #    'begin':   lambda *x: x[-1],
    #    'car':     lambda x: x[0],
    #    'cdr':     lambda x: x[1:], 
    #    'cons':    lambda x,y: [x] + y,
    #    'eq?':     op.is_, 
    #    'expt':    pow,
    #    'equal?':  op.eq, 
    #    'length':  len, 
    #    'list':    lambda *x: List(x), 
    #    'list?':   lambda x: isinstance(x, List), 
    #    'map':     map,
    #    'max':     max,
    #    'min':     min,
    #    'not':     op.not_,
    #    'null?':   lambda x: x == [], 
    #    'number?': lambda x: isinstance(x, Number),  
    #	'print':   print,
    #    'procedure?': callable,
    #    'round':   round,
    #    'symbol?': lambda x: isinstance(x, Symbol),
    #})

    return env

def evaluate(x, env=None): #TODO: add sigma, or something
    if env is None:
        env = standard_env()
    if isinstance(x, Symbol):    # variable reference
        if x.startswith('"') and x.endswith('"'):
            return x
        else:
            return env[x] #env.find(x)[x]
    elif not isinstance(x, List):# constant 
        return torch.FloatTensor([x])
    op, *args = x       
    if op == 'if':             # conditional
        (test, conseq, alt) = args
        exp = (conseq if evaluate(test, env) else alt)
        return evaluate(exp, env)
    elif op == 'fn':             # procedure
        (parms, body) = args
        return Procedure(parms, body, env)
    elif op == 'push-address':
        (addr, body) = args
        return env['push-address'](addr, body)
    else:                        # procedure call
        proc = evaluate(op, env)
        vals = [evaluate(arg, env) for arg in args]
        return proc(*vals[1:])


def get_stream(exp):
    while True:
        yield evaluate(exp).call()


def run_deterministic_tests():
    
    for i in range(1,14):

        exp = daphne(['desugar-hoppl', '-i', '/home/bigboi/Desktop/Prob_Prog_Course/CS532-HW5/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        print(i, exp)
        ret = evaluate(exp).call()
        print("truth", truth, "return", ret)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('FOPPL Tests passed')
        
    for i in range(1,13):

        exp = daphne(['desugar-hoppl', '-i', '/home/bigboi/Desktop/Prob_Prog_Course/CS532-HW5/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        print(i, exp)
        ret = evaluate(exp).call()
        print("truth", truth, "return", ret)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-2
    
    for i in range(1,7):
        exp = daphne(['desugar-hoppl', '-i', '/home/bigboi/Desktop/Prob_Prog_Course/CS532-HW5/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        print("exp", exp, "truth", truth)
        stream = get_stream(exp)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    



if __name__ == '__main__':
    
    run_deterministic_tests()
    run_probabilistic_tests()
    
    print(int('a'))

    samples = {}
    for i in range(1,4):
        print('processing', i)
        exp = daphne(['desugar-hoppl', '-i', '/home/bigboi/Desktop/Prob_Prog_Course/CS532-HW5/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        samples[i] = []
        for j in range(1000):
            print("Sample", j)
            samples[i].append(evaluate(exp).call())

with open("samples.pk", "wb+") as f:
    pickle.dump(samples, f)      
