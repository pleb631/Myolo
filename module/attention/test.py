from thop import clever_format
from thop import profile
import torch

#from SPPF import SPPF as net
import time

test_modules = ['SOCA']


for unit in test_modules:
    print('-'*20,f"\ntesting {unit} ......")
    module =  __import__(unit)
    net =  getattr(module,unit)
    model = net()
    input = torch.randn(1, 512, 14, 14)
    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print(f'flop:{flops}\nparams:{params}')
    
    model = model#.cuda()
    input = input#.cuda()
    for i in range(100):
        _=model(input)
    
    s = time.time()
    for i in range(1000):
        _=model(input)
    e = time.time()
    print(f'fps:{e-s} ms')
    print('-'*20)