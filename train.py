from robustness import model_utils, datasets
import torch
import wandb
import sys
import pytorch_ssim

# Constants
DATA = 'CIFAR' # Choices: ['CIFAR', 'ImageNet', 'RestrictedImageNet']
BATCH_SIZE = 50
NUM_WORKERS = 8
NUM_CLASSES_VIS = 10

DATA_SHAPE = 32 # Image size (fixed for dataset)
REPRESENTATION_SIZE = 2048 # Size of representation vector (fixed for model)
GRAIN = 1

USE_WANDB = True
DISPLAY_COUNT = 16

# Load dataset
dataset_function = getattr(datasets, DATA)
dataset = dataset_function('./datasets/CIFAR10')
train_loader, test_loader = dataset.make_loaders(workers=NUM_WORKERS, batch_size=BATCH_SIZE, data_aug=False)
data_iterator = enumerate(train_loader) 

# Loss function for image generation procedure
def cross_entropy(mod, inp, targ):
    op = mod(inp)
    loss = torch.nn.CrossEntropyLoss(reduction='none')(op, targ)
    return loss, None

# Loss function used to evalaute the quality of reconstructions
#gen_loss = torch.nn.MSELoss()
gen_loss = torch.nn.L1Loss()
'''
ssim_loss = pytorch_ssim.SSIM()
def gen_loss(img1, img2):
    ssim_out = -ssim_loss(img1, img2)
    ssim_value = - ssim_out.data[0]
    return ssim_value
'''

def train(epochs, learning_rate, mom, eps, step_size, wd, iterations, od):
    # Load model to transfer
    attacker_model, _ = model_utils.make_and_restore_model(
        arch='resnet50',
        dataset=dataset,
        resume_path='./robust_models/CIFAR_model.pt'
    )
    attacker_model = attacker_model.to('cuda:0')

    # Declare the optimiser for the model's parameters
    #param_optim = torch.optim.Adam(attacker_model.model.parameters(), lr=learning_rate) #lr=0.1
    param_optim = torch.optim.SGD(attacker_model.model.parameters(), lr=learning_rate, momentum=mom, weight_decay=wd)

    # not important
    target = torch.zeros((BATCH_SIZE, )).to('cuda:0').long()

    # TRAINING LOOP
    epoch = 0
    while epoch < epochs:
        for im_seed in train_loader:
            # Get next batch of images
            im_seed = im_seed[0].to('cuda:0').float()
            im_seed.requires_grad = True

            # quick sanity check for type/shape
            if epoch == 0:
                print(type(im_seed))
                print(im_seed.size())

            param_optim.zero_grad()

            # Arguments for the attacker model
            attacker_kwargs = {
                'custom_loss': cross_entropy,
                'constraint':'2',
                'eps': eps,
                'step_size': step_size,
                'iterations': iterations,
                'targeted': False,
            }  

            # Run the attacker model
            _, im_gen = attacker_model(im_seed, target, make_adv=True, **attacker_kwargs)
            im_gen = im_gen.to('cuda:0').float()
            im_gen.requires_grad = True

            # Compute the loss and backpropagate
            l = gen_loss(im_seed, im_gen)
            l.backward()
            param_optim.step()

            # Print to stdout every 100 epochs
            if epoch % 100 == 0:
                print(f"epoch: {epoch}, loss: {l}")

            # If we're using wandb send updates every 200 epochs
            if USE_WANDB and (epoch % 200 == 0):
                # Correctly format the seeds and generated images
                permuted_im_seed = torch.permute(im_seed, (0, 2, 3, 1))[0:DISPLAY_COUNT]
                numpy_im_seed = permuted_im_seed.cpu().detach().numpy()
                images_im_seed = [ wandb.Image(i, caption="seed") for i in numpy_im_seed ] 

                permuted_im_gen = torch.permute(im_gen, (0, 2, 3, 1))[0:DISPLAY_COUNT]
                numpy_im_gen = permuted_im_gen.cpu().detach().numpy()
                images_im_gen = [ wandb.Image(i, caption="generated") for i in numpy_im_gen ] 

                # Populate the images array
                images = []
                for i in range(DISPLAY_COUNT*2):
                    if i % 2 == 0:
                        images.append(images_im_seed[i // 2])
                    else:
                        images.append(images_im_gen[i // 2])

                # Send to wandb
                wandb.log({
                    "epoch": epoch, 
                    "loss": l,
                    "images": images,
                })

            # Save the model every thousand epochs (overwrites the previous model at the output location)
            if epoch % 1000 == 0:
                torch.save(attacker_model, od)
                
            epoch += 1

    return attacker_model

if __name__ == "__main__":
    # Set device to gpu
    torch.cuda.device(0)

    # Format output dir (read cmd line args to do this)
    output_dir = f"./robust_models/trained_model_{sys.argv[1]}.pt"
    print(f"Model will be saved to {output_dir}")

    # HYPERPARAMETERS
    lr = 0.05
    mom = 0.9
    epochs = 10000
    eps = 200
    step_size = 5
    iterations = 10
    weight_decay = 0

    # Init wandb
    if USE_WANDB:
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
                "iterations": iterations,
                "weight_decay": weight_decay
            },

            # Set the name of the run
            name=f"{sys.argv[1]}_transfer_image_gen",
        )

    # Train the nn and save it
    final_model = train(epochs, lr, mom, eps, step_size, weight_decay, iterations, output_dir)
    torch.save(final_model, output_dir)