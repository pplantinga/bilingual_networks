import torch
from tqdm import tqdm # for displaying progress bar
import os
import numpy as np
# from data import CallFriend, read_config
import models
import pandas as pd
from glob import glob
from copy import deepcopy
import sys
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        if isinstance(self.model, getattr(models, config.training_type)):
            self.lr = config.training_lr
            self.checkpoint_path = os.path.join(self.config.folder, "training")
        else:
            self.lr = config.pretraining_lr
            self.checkpoint_path = os.path.join(self.config.folder, "pretraining")

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.epoch = 0
        self.df = None

    def load_checkpoint(self, i_checkpoint=None):
        if i_checkpoint is None:
            state_file = "model_state.pth"
        else:
            state_file = "model_state_{}.pth".format(i_checkpoint)
        if os.path.isfile(os.path.join(self.checkpoint_path, state_file)):
            try:
                if self.model.is_cuda:
                    if i_checkpoint is None:
                        self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_path, state_file)))
                    else:
                        self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_path, state_file)))
                else:
                    self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_path, state_file), map_location="cpu"))
            except:
                print("Could not load previous model; starting from scratch")
        else:
            print("No previous model; starting from scratch")

    def save_checkpoint(self, i_checkpoint=None):
        try:
            if i_checkpoint is None:
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, "model_state.pth"))
            else:
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, "model_state_{}.pth".format(i_checkpoint)))
        except:
            print("Could not save model")

    def log(self, results):
        if self.df is None:
            self.df = pd.DataFrame(columns=[field for field in results])
        self.df.loc[len(self.df)] = results
        self.df.to_csv(os.path.join(self.checkpoint_path, "log.csv"))

    def generate_inputs_and_targets(self, batch, lang_input=1, phone_target=1, word_target=1):
        """
        lang_input
        0: no language input
        1: language input half of the time
        2: language input always
        phone_target
        0: no phoneme targets
        1: concatenate phoneme inventories of different languages to use as targets
        2: superset of phonemes (combine homologues across languages) as targets
        word_target
        0: no word targets
        1: concatenate vocabularies of different languages to use as targets
        """
        x,y,lengths = batch
        if self.model.is_cuda:
            torch.cuda.empty_cache()
            x = x.cuda()
            y = tuple([yy.cuda() for yy in y])
            lengths = lengths.cuda()
        spec = self.model.feat(x[:,:-1])
        lengths = ((lengths-1)/self.model.downsample_factor).long()
        y_lang = deepcopy(y[0][:,0])
        if lang_input==0: # Do not provide language tags in the input
            y_lang_in = torch.ones_like(y_lang)*3
        elif lang_input==1: # Provide language tags only half of the time
            is_langin = torch.rand(y_lang.shape)>.5
            y_lang_in = torch.ones_like(y_lang)*3
            y_lang_in[is_langin] = y_lang[is_langin]
        elif lang_input==2: # Always provide language tags
            y_lang_in = y_lang
        if type(self.model).__name__=='ASRSpecNet':
            if phone_target==0:
                y[3].fill_(0)
                y[1].fill_(0)
            elif phone_target==1:
                y[3].fill_(0)
            elif phone_target==2:
                y[1].fill_(0)
            if word_target==0:
                y[2].fill_(0)
            targets = [y[1], y[2], y[3]] # phonemes, words, homologes
        elif type(self.model).__name__=='ASRSpecNetPhonemes':
            targets = [y[1]] # phonemes
        inputs = [spec, y_lang_in]
        return inputs, targets, lengths

    def train(self, dataset, print_interval=100, tqdmout=sys.stderr):
        train_losses = [0 for i_loss in range(len(self.model.losses))]
        num_examples = 0
        self.model.train()
        for idx, batch in enumerate(tqdm(dataset.loader,file=tqdmout)):
            inputs, targets, lengths = self.generate_inputs_and_targets(
                batch, self.config.pretraining_langin, self.config.pretraining_phoneout,
                self.config.pretraining_wordout
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.config.use_amp):
                outputs = self.model(inputs, lengths)
                loss, losses, accuracies = self.model.criterion(outputs, targets, lengths)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            batch_size = inputs[0].shape[0]
            num_examples += batch_size
            for i_loss in range(len(self.model.losses)):
                train_losses[i_loss] += losses[i_loss].cpu().data.numpy().item() * batch_size
            if idx % print_interval == 0:
                msg = "\ntrain loss:"
                for i_loss in range(len(losses)):
                    msg += " %s: %.5f accuracy %.2f" %(self.model.losses[i_loss],
                                               losses[i_loss].cpu().data.numpy().item(),
                                               accuracies[i_loss].cpu().data.numpy().item())
                print(msg+"\n")
        for i_loss in range(len(self.model.losses)):
            train_losses[i_loss] = train_losses[i_loss]/num_examples
            results = {"train_loss" : train_losses[i_loss], "set": 'train'}
            self.log(results)
        self.epoch += 1
        return train_losses

    def test(self, dataset, set='valid', h0=None, tqdmout=sys.stderr):
        test_losses = [0 for i_loss in range(len(self.model.losses))]
        test_accuracy = [0 for i_loss in range(len(self.model.losses))]
        num_samples = 0
        num_examples = 0
        soft = torch.nn.Softmax(dim=2)
        self.model.eval()
        for idx, batch in enumerate(tqdm(dataset.loader,file=tqdmout)):
            inputs, targets, lengths = self.generate_inputs_and_targets(
                batch, self.config.pretraining_langin_test, self.config.pretraining_phoneout,
                self.config.pretraining_wordout
            )
            outputs = self.model(inputs, lengths, h0=h0)
            loss, losses, accuracies = self.model.criterion(outputs, targets, lengths)
            batch_size = len(lengths)
            num_examples += batch_size
            num_samples += int(sum(lengths))
            for i_loss in range(len(self.model.losses)):
                test_losses[i_loss] += losses[i_loss].cpu().data.numpy().item() * batch_size
                test_accuracy[i_loss] += accuracies[i_loss].cpu().data.numpy().item() * batch_size
        for i_loss in range(len(self.model.losses)):
            test_losses[i_loss] = test_losses[i_loss]/num_examples
            test_accuracy[i_loss] = test_accuracy[i_loss]/num_examples
            results = {"train_loss" : test_losses[i_loss], "set": set}
            self.log(results)
        return test_losses, test_accuracy
