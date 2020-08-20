import os
import shutil
import math
import random
import time
from datetime import timedelta
import torch
from torch.nn import functional as F
from torch.distributions import kl_divergence, Normal
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.metrics import RunningAverage, Average
from bvae.model import BetaVAE

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
random.seed(27)
torch.manual_seed(27)
if cuda: torch.cuda.manual_seed(27)


def train(run_name, train_set, test_set,
          n_iterations, batch_size, lr, beta, z_dim):
    """
    This generically handles the training loop for all models and datasets. It uses
    the Ignite package in order to simplify this process and add useful features.
    Among other things, it:
    - Instantiates the data loaders, optimizer, loss
    - Specifies  the forward/backward pass for training and evaluation
    - Keeps track of training/test metrics and logs them to tensorboard
    - Saves the model with the lowest test loss
    :param run_name: Model/logs will be saved under bvae/saved_runs/[run_name]
    :param train_set: Instance of a Dataset defined under the 'dataset.py' file
    :param test_set: Instance of a Dataset defined under the 'dataset.py' directory
    :param n_iterations: Number of training iterations
    :param batch_size: Training and test batch size
    :param lr: Learning rate
    :param beta: Beta parameter for disentangling
    """
    # Make the run directory
    if not os.path.exists('bvae/saved_runs'):
        os.mkdir('bvae/saved_runs')
    save_dir = os.path.join('bvae/saved_runs', run_name)
    if run_name == 'debug':     # If we're debugging, just remove the old run without returning an error
        shutil.rmtree(save_dir, ignore_errors=True)
    os.mkdir(save_dir)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))      # Tensorboard logging

    # Instantiate model, data loaders, loss function, and optimizer
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    model = BetaVAE(z_dim=z_dim, nc=train_set.nc).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training step callback
    def train_step(engine, images):
        model.train()
        images = images.to(device)
        recon, mu, logvar = model(images)
        loss, recon_loss, kl = loss_func(images, recon, mu, logvar, beta, model.decoder_distribution)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {'beta_loss': loss, 'recon_loss': recon_loss, 'kl': kl}

    # Evaluation step callback
    def eval_step(engine, images):
        model.eval()
        with torch.no_grad():
            images = images.to(device)
            recon, mu, logvar = model(images)
            loss, recon_loss, kl = loss_func(images, recon, mu, logvar, beta, model.decoder_distribution)
        return {'beta_loss': loss, 'recon_loss': recon_loss, 'kl': kl}

    # Ignite train/evaluation engines
    train_engine = Engine(train_step)
    test_engine = Engine(eval_step)

    # Train/test metrics
    RunningAverage(output_transform=lambda x: x['beta_loss']).attach(train_engine, 'Beta Loss')
    RunningAverage(output_transform=lambda x: x['recon_loss']).attach(train_engine, 'Reconstruction Loss')
    RunningAverage(output_transform=lambda x: x['kl']).attach(train_engine, 'KL Divergence')
    Average(output_transform=lambda x: x['beta_loss']).attach(test_engine, 'Beta Loss')
    Average(output_transform=lambda x: x['recon_loss']).attach(test_engine, 'Reconstruction Loss')
    Average(output_transform=lambda x: x['kl']).attach(test_engine, 'KL Divergence')

    # Progress bar displaying training loss and accuracy
    ProgressBar(persist=True).attach(train_engine, metric_names=['Beta Loss', 'Reconstruction Loss', 'KL Divergence'])

    # Model checkpointing (keep model with lowest test loss)
    checkpoint_handler = ModelCheckpoint(os.path.join(save_dir, 'checkpoints'), type(model).__name__,
                                         score_function=lambda eng: -eng.state.metrics['Beta Loss'])
    test_engine.add_event_handler(event_name=Events.COMPLETED, handler=checkpoint_handler, to_save={'model': model})

    # Early stopping if the test set loss does not decrease over 5 epochs
    early_stop_handler = EarlyStopping(patience=5, trainer=train_engine,
                                       score_function=lambda eng: -eng.state.metrics['Beta Loss'])
    test_engine.add_event_handler(Events.COMPLETED, early_stop_handler)

    # Log training metrics to tensorboard every 100 batches
    @train_engine.on(Events.ITERATION_COMPLETED(every=100))
    def log_training_metrics(engine):
        for metric, value in engine.state.metrics.items():
            writer.add_scalar('training/{}'.format(metric), value, engine.state.iteration)

    # Print and log test metrics to tensorboard after every epoch
    @train_engine.on(Events.EPOCH_COMPLETED)
    def log_test_metrics(engine):
        test_engine.run(test_loader)
        results = ['Test Results - Epoch: {}'.format(engine.state.epoch)]
        for metric, value in test_engine.state.metrics.items():
            writer.add_scalar('test/{}'.format(metric), value, engine.state.iteration)
            results.append('{}: {:.2f}'.format(metric, value))
        print(' '.join(results))


    # Save some sample images
    @train_engine.on(Events.EPOCH_COMPLETED)
    def log_sample_images(engine):
        train_samples = torch.stack([train_set[i] for i in range(min(36, len(train_set)))]).to(device)
        test_samples = torch.stack([test_set[i] for i in range(min(36, len(train_set)))]).to(device)
        with torch.no_grad():
            train_recon, _, _ = model(train_samples)
            test_recon, _, _ = model(test_samples)
        train_recon = train_recon.cpu()
        test_recon = test_recon.cpu()
        writer.add_image('training/Original', make_grid(train_samples, nrow=6), engine.state.iteration)
        writer.add_image('training/Reconstructed', make_grid(train_recon, nrow=6), engine.state.iteration)
        writer.add_image('test/Original', make_grid(test_samples, nrow=6), engine.state.iteration)
        writer.add_image('test/Reconstructed', make_grid(test_recon, nrow=6), engine.state.iteration)


    # Gracefully terminate on any exception, and simply end training + save the current model
    # if we are manually stopping with a keyboard interrupt (e.g. model was converging)
    @train_engine.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        writer.close()
        engine.terminate()
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            import warnings
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')
            checkpoint_handler(test_engine, {'model_exception': model})
        else:
            raise e

    # Start off training and report total execution time when over
    start_time = time.time()
    n_epochs = math.ceil(n_iterations / (len(train_set) / batch_size))
    train_engine.run(train_loader, n_epochs)
    writer.close()
    end_time = time.time()
    print('Total training time: {}'.format(timedelta(seconds=end_time - start_time)))


def loss_func(x, x_recon, mu, logvar, beta, decoder_distribution):
    if decoder_distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum') / x.size(0)
    elif decoder_distribution == 'gaussian':
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
    else:
        raise NotImplementedError('Unimplemented decoder distribution type: {}'.format(decoder_distribution))

    p = Normal(torch.zeros_like(mu), torch.ones_like(mu))
    q = Normal(mu, torch.exp(0.5 * logvar))
    kl = kl_divergence(p, q).sum(dim=1).mean()

    beta_loss = recon_loss + beta * kl

    return beta_loss, recon_loss, kl
