import torch
import jittor as jt
clip = torch.load('ViT-B-16.pt').state_dict()

for k in clip.keys():
    clip[k] = clip[k].float().cpu()
jt.save(clip, 'ViT-B-16.pkl')