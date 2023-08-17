from robustness import model_utils, datasets
import torch
import wandb
import sys

# Constants
DATA = 'CIFAR' # Choices: ['CIFAR', 'ImageNet', 'RestrictedImageNet']
BATCH_SIZE = 100
NUM_WORKERS = 8
NUM_CLASSES_VIS = 10

DATA_SHAPE = 32 # Image size (fixed for dataset)
REPRESENTATION_SIZE = 2048 # Size of representation vector (fixed for model)
GRAIN = 1

# Load dataset
dataset_function = getattr(datasets, DATA)
dataset = dataset_function('./datasets/CIFAR10')
train_loader, test_loader = dataset.make_loaders(workers=NUM_WORKERS, batch_size=BATCH_SIZE, data_aug=False)
data_iterator = enumerate(train_loader) 

def cross_entropy(mod, inp, targ):
    op = mod(inp)
    loss = torch.nn.CrossEntropyLoss(reduction='none')(op, targ)
    return loss, None

gen_loss = torch.nn.MSELoss()

def train(epochs, learning_rate, mom, eps, step_size, iterations):
    attacker_model, _ = model_utils.make_and_restore_model(
        arch='resnet50',
        dataset=dataset,
        resume_path='./robust_models/CIFAR_model.pt'
    )

    attacker_model = attacker_model.to('cuda:0')

    images = torch.randn((BATCH_SIZE, 3, 32, 32)).to('cuda:0')

    param_optim = torch.optim.SGD(attacker_model.model.parameters(), lr=learning_rate, momentum=mom)
    image_optim = torch.optim.SGD([images], lr=learning_rate, momentum=mom)

    target = torch.zeros((BATCH_SIZE, )).to('cuda:0').long()

    epoch = 0
    while epoch < epochs:
        param_optim.zero_grad()
        image_optim.zero_grad()

        attacker_kwargs = {
            'custom_loss': cross_entropy,
            'constraint':'2',
            #'eps': 30,
            'eps': eps,
            #'step_size': 1,
            'step_size': step_size,
            #'iterations': 10,
            'iterations': iterations,
            'targeted': False,
        }  

        _, im_gen = attacker_model(images, target, make_adv=True, **attacker_kwargs)
        im_gen = im_gen.to('cuda:0').float()
        im_gen.requires_grad = True

        l = gen_loss(images, im_gen)
        l.backward()
        param_optim.step()
        image_optim.step()

        if epoch % 50 == 0:
            wandb.log({"epoch": epoch, "loss": l})
            print(f"epoch: {epoch}, loss: {l}")

        epoch += 1

    return attacker_model, images

if __name__ == "__main__":
    torch.cuda.device(0)

    lr = 0.001
    mom = 0.9
    epochs = 5000
    eps = 200
    step_size = 5
    iterations = 30

    run = wandb.init(
        # Set the project where this run will be logged
        project="Dataset_Extraction",

        # Track hyperparameters and run metadata
        config={
            "lr": lr,
            "momentum": mom,
            "epochs": epochs,
            "epsilon": eps,
            "step_size": step_size,
            "iterations": iterations
        }
    )

    final_model, final_images = train(epochs, lr, mom, eps, step_size, iterations)

    torch.save(final_model, f"./robust_models/trained_model_{sys.argv[1]}.pt")
    torch.save(final_images, f"./robust_models/trained_images_{sys.argv[1]}.pt")

    