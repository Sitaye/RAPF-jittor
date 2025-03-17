# JCLIP

[[Blog]](https://openai.com/blog/clip/) [[Paper]](https://arxiv.org/abs/2103.00020)







### environment installation
```bash
pip install jittor
pip install ftfy regex tqdm
python setup.py develop
```

### Model weights

download[VIT-B-32](https://github.com/uyzhang/JCLIP/releases/tag/%E6%9D%83%E9%87%8D) or using python code:

```python
import torch
import jittor as jt
clip = torch.load('ViT-B-32.pt').state_dict()

for k in clip.keys():
    clip[k] = clip[k].float().cpu()
jt.save(clip, 'ViT-B-32.pkl')
```


### demo
```python
import jittor as jt
import jclip as clip
from PIL import Image

jt.flags.use_cuda = 1

model, preprocess = clip.load("ViT-B-32.pkl")

image = preprocess(Image.open("CLIP.png")).unsqueeze(0)

text = clip.tokenize(["a diagram", "a dog", "a cat"])

with jt.no_grad():
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
```



