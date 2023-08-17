import torch
from torch.distributions.multivariate_normal import MultivariateNormal

DATA_SHAPE = 32 # Image size (fixed for dataset)
REPRESENTATION_SIZE = 2048 # Size of representation vector (fixed for model)
GRAIN = 1

def downsample(x, step=GRAIN):
    down = torch.zeros([len(x), 3, DATA_SHAPE//step, DATA_SHAPE//step])

    for i in range(0, DATA_SHAPE, step):
        for j in range(0, DATA_SHAPE, step):
            v = x[:, :, i:i+step, j:j+step].mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            ii, jj = i // step, j // step
            down[:, :, ii:ii+1, jj:jj+1] = v
    return down

def upsample(x, step=GRAIN):
    up = torch.zeros([len(x), 3, DATA_SHAPE, DATA_SHAPE])

    for i in range(0, DATA_SHAPE, step):
        for j in range(0, DATA_SHAPE, step):
            ii, jj = i // step, j // step
            up[:, :, i:i+step, j:j+step] = x[:, :, ii:ii+1, jj:jj+1]
    return up

def distribution(target_index, test_loader):
    im_test, targ_test = [], []
    for _, (im, targ) in enumerate(test_loader):
        im_test.append(im)
        targ_test.append(targ)
    im_test, targ_test = torch.cat(im_test), torch.cat(targ_test)

    imc = im_test[targ_test == target_index]
    down_flat = downsample(imc).view(len(imc), -1)
    mean = down_flat.mean(dim=0)
    down_flat = down_flat - mean.unsqueeze(dim=0)
    cov = down_flat.t() @ down_flat / len(imc)
    dist = MultivariateNormal(mean, covariance_matrix=cov+1e-4*torch.eye(3 * DATA_SHAPE//GRAIN * DATA_SHAPE//GRAIN))

    return dist