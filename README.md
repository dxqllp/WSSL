#  WSSL++: Redesigning weak semi-supervised learning for polyp segmentation


##  Requirements

* torch
* torchvision 
* tqdm
* opencv
* scipy
* skimage
* PIL
* numpy

### step1. Training teacher

```bash
python main.py  --train_per '50per'/

```

###  step2. Inference pseudo label

```bash
python test.py  
```
###  step3. Training student
```bash
python train_stu.py  
```

