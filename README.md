## Description:
Consider a robust classifier for $d$-dimensional images with $p$ parameters. We define $$\psi : \mathbb{R}^d \times \mathbb{R^p} \rightarrow \mathbb{R}^d$$ to be the result of the pertubation algorithm on that robust classifier as per [MadryLab/robustness_applications](https://github.com/MadryLab/robustness_applications) (i.e. $\psi(x_i; \theta)$ is the result of running the pertubation algorithm from a start point $x_i$ with model parameters $\theta$).

We then define the following loss function across the parameters of our robust classifier $\theta \in \mathbb{R}^p$ and the elements of our training set $\lbrace x_i \rbrace _ {i=1}^n \subset \mathbb{R}^d$,

$$L(\theta, \lbrace x_i \rbrace _ {i=1}^n) = \frac{1}{n}\sum_{i = 1}^n ||x _i - \psi(x _i; \theta)||^2$$

where $||\cdot||$ is the Euclidean norm (i.e. this loss function is the Mean Squared Error function between the start and end points of the pertubation algorithm). We then compute $\tilde{\theta} = \underset{\theta}{\textrm{argmin  }}  L(\theta, \{x_i\}_{i=1}^n)$ using stochastic gradient descent to find better model parameters for image generation. 

## Dependencies:
- Pytorch
- wandb (if you don't want to use this change the ```USE_WANDB``` flag in ```./train.py```)
- [Robustness](https://github.com/MadryLab/robustness) (install this directly from github using ```pip install git+https://github.com/MadryLab/robustness```)

I may have forgot something but these libraries + their dependencies *should* be all you need.

## Usage
1. Download [this CIFAR model](http://andrewilyas.com/CIFAR.pt) and put it at ```./robust_models/CIFAR_model.pt```
2. Run ```python train.py <MODEL_NAME>``` to transfer the robust classifier according to the new loss function (this should install CIFAR in ```./datasets```)
3. Use ```./visualise_generation.ipynb``` to generate some images with your new classifier

Included is a pretrained classifier ```./robust_models/trained_model_1.pt``` if you can't be bothered to train your own.
