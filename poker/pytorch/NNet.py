import os
import sys
import time
import numpy as np
from tqdm import tqdm
import wandb # Add wandb import
import math # Add math import

sys.path.append(os.environ["BASE_DIR"] + "/alpha-poker")
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR # Add LambdaLR import

from poker.pytorch.PokerNNet import PokerNNet
from poker.PokerLogic import PokerState

args = dotdict({
    'lr': 0.0005,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'block_width': 256,
    'n_blocks': 3,
    # 'run_name': 'poker_run_' + str(int(time.time())) # Remove run_name, wandb initialized externally
})

class NNetWrapper(NeuralNet):
    def __init__(self, game, a=None):
        global args
        if a:
            args = a

        print("Args: ", args)
        # Remove wandb.init() - Assume it's initialized externally
        if args.use_wandb:
            wandb.init(project="alpha-poker", name=args.wandb_run_name, config=args, reinit=True)

        self.input_size = len(PokerState().to_vector())  # PokerState.to_vector() length
        self.action_size = 4
        self.nnet = PokerNNet(self.input_size, self.action_size, args)
        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)

        # Calculate total steps and warmup steps
        batch_count = int(len(examples) / args.batch_size)
        total_steps = args.epochs * batch_count
        warmup_epochs = 3
        warmup_steps = warmup_epochs * batch_count
        final_lr_factor = 0.1 # End at 10% of initial LR

        # Define the LR scheduler function
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay phase
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                # Decay from 1.0 down to final_lr_factor
                decayed_factor = (1 - final_lr_factor) * cosine_decay + final_lr_factor
                return decayed_factor

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            total_losses = AverageMeter() # Add total loss meter
            # batch_count calculated above
            print(f"Batch count: {batch_count}")
            t = tqdm(range(batch_count), desc='Training Net')
            for batch_idx in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                states, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                # SAM: I changed this to use PokerState.to_vector()
                states = torch.FloatTensor(np.array([state.to_vector() for state in states]).astype(np.float32))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float32))
                if args.cuda:
                    states, target_pis, target_vs = states.cuda(), target_pis.cuda(), target_vs.cuda()
                out_pi, out_v = self.nnet(states)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # Log batch losses and learning rate to wandb
                current_lr = optimizer.param_groups[0]['lr']
                if args.use_wandb:
                    wandb.log({
                        "batch_pi_loss": l_pi.item(),
                        "batch_v_loss": l_v.item(),
                        "batch_total_loss": total_loss.item(),
                        "learning_rate": current_lr, # Log LR
                        "epoch": epoch,
                        "batch_idx": batch_idx
                    })

                pi_losses.update(l_pi.item(), states.size(0))
                v_losses.update(l_v.item(), states.size(0))
                total_losses.update(total_loss.item(), states.size(0)) # Update total loss meter
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses, LR=current_lr) # Show LR in tqdm
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step() # Step the scheduler after the optimizer

            if args.use_wandb:
                # Log epoch average losses
                wandb.log({
                    "epoch_pi_loss": pi_losses.avg,
                    "epoch_v_loss": v_losses.avg,
                    "epoch_total_loss": total_losses.avg,
                    "epoch": epoch
                })

    def predict(self, state):
        # state: np array (vectorized PokerState)
        self.nnet.eval()
        state = torch.FloatTensor(state.to_vector().astype(np.float32))
        if args.cuda:
            state = state.cuda()
        state = state.view(1, -1)
        with torch.no_grad():
            pi, v = self.nnet(state)
        
        # Need to scale back to actual values since rewards were scaled
        scaled_v = v.cpu().numpy()[0] * 1000.0
        return torch.exp(pi).cpu().numpy()[0], scaled_v

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(f"Checkpoint Directory does not exist! Making directory {folder}")
            os.mkdir(folder)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception(f"No model in path {filepath}")
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
