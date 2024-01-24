import matplotlib.image as mpimg
import numpy as np
with open('WHU-CD-256/list/train.txt','r') as f:
    data = f.readlines()

with open('WHU-CD-256/list/train_label.txt','w') as f:
    for fname in data:
        image = mpimg.imread(f'WHU-CD-256/label/{fname}'[:-1])
        if np.all(image==0):
            f.write(f'{fname[:-1]},0,0,0\n')
        else:
            f.write(f'{fname[:-1]},0,0,1\n')