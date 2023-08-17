## Dependencies:
- Pytorch
- wandb (if you don't want to use this change the ```USE_WANDB``` flag in ```./train.py```)
- [Robustness](https://github.com/MadryLab/robustness) (install this directly from github using ```pip install git+https://github.com/MadryLab/robustness```)

I may have forgotten something from this but this *should* be all you need.

## Usage
1. Download [this CIFAR model](http://andrewilyas.com/CIFAR.pt) and put it at ```./robust_models/CIFAR_model.pt```
2. Run ```python train.py <MODEL_NAME>``` to transfer the robust classifier according to the new loss function (this should install CIFAR in ```./datasets```)
3. Use ```visualise_generation.ipynb``` to generate some images with your new classifier

Included is a pretrained classifier ```./robust_models/trained_model_1.pt``` if you can't be bothered to train your own.