import torch
import sys
import collections

model = torch.load(sys.argv[1], map_location='cpu')
new_d = collections.OrderedDict()

for k, v in model.items():
    if k == 'joint.ffn_out.weight':
        new_d['joint.ffn_out_ori.weight'] = v.clone()
    if k == 'joint.ffn_out.bias':
        new_d['joint.ffn_out_ori.bias'] = v.clone()
    new_d[k] = v
        
torch.save(new_d, sys.argv[2])
