import torch
from torch.optim import lr_scheduler
import numpy as np
from data import read_config, get_datasets, get_generic_dataset
import models
from training import Trainer
import argparse
import sys, os, json
from word_phone_model import WordPhoneModel
import torchaudio_contrib as tac


def parse_args():

    # Get args
    parser = argparse.ArgumentParser()
    parser.add_argument('--restart', action='store_true', help='only test')
    parser.add_argument('--pretrain', action='store_true', help='only test')
    parser.add_argument('--test', action='store_true', help='only test')
    parser.add_argument('--train', action='store_true', help='only test')
    parser.add_argument('--train_test', action='store_true', help='only test')
    parser.add_argument('--is_cpu', action='store_true', help='run on CPU instead')
    parser.add_argument('--datapath', type=str, default='/cs/department2/data/commonvoice/')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loading')
    parser.add_argument('--config_path', type=str,
                        help='path to config file with hyperparameters, etc.')
    parser.add_argument('--num_epochs', type=int, default=None, help='')
    args = parser.parse_args()

    return args

def setup_pretraining(config, datapath, num_workers, is_cpu):

    train_dataset, valid_dataset, test_dataset = get_datasets(config, datapath,
        config.pretraining_manifest_train,
        config.pretraining_manifest_dev,
        config.pretraining_manifest_test, num_workers)

    # Initialize model
    #model = getattr(models, config.type)(config=config)
    model = WordPhoneModel(
        languages=5,
        input_size=config.n_mel,
        word_outputs=config.vocabulary_size,
        phone_outputs=config.num_phonemes,
        homolog_outputs=config.num_homologes,
        cnn_channels=config.cnn_out_size,
        rnn_layers=config.num_rnn_layers,
        rnn_dropout=config.rnn_drop,
        rnn_bidirectional=config.rnn_bidirectional,
    ).to("cuda")
    model.criterion = models.compute_criterion
    model.losses = ['phonemes', 'words', 'homologes']
    model.is_cuda = True
    mel = tac.layers.Melspectrogram(
        config.n_mel, config.fs, fft_length=2**10, hop_length=config.downsample_factor
    )
    model.feat = torch.nn.Sequential(mel, models.AmplitudeToDb()).to("cuda")
    model.downsample_factor = config.downsample_factor

    if is_cpu:
        model.is_cuda = False
        model.cpu()

    # Train the model
    trainer = Trainer(model=model, config=config)

    return trainer, train_dataset, valid_dataset, test_dataset

def setup_training(config, datapath, num_workers, is_cpu):

    config.pretraining_batch_size = config.training_batch_size
    train_dataset, valid_dataset, test_dataset = get_datasets(config, datapath,
        config.training_manifest_train,
        config.training_manifest_dev,
        config.training_manifest_test, num_workers)

    # Initialize model
    model = getattr(models, config.training_type)(config=config)
    if is_cpu:
        model.is_cuda = False
        model.cpu()

    # Train the model
    trainer = Trainer(model=model, config=config)

    return trainer, train_dataset, valid_dataset, test_dataset

if __name__ == "__main__":

    args = parse_args()

    pretrain = args.pretrain
    train = args.train
    test = args.test
    train_test = args.train_test
    restart = args.restart
    config_path = args.config_path
    is_cpu = args.is_cpu
    datapath = args.datapath
    num_workers = args.num_workers
    num_epochs = args.num_epochs

    # Read config file
    config = read_config(config_path)
    config.datapath = datapath
    if num_epochs is not None:
        config.pretraining_num_epochs = num_epochs

    if pretrain or test:
        trainer, train_dataset, valid_dataset, test_dataset = setup_pretraining(config, datapath, num_workers, is_cpu)

    if pretrain:

        if restart:
            trainer.load_checkpoint() # test with best model (acc. to validation)
        print('Learning rate: %.10f' %trainer.optimizer.param_groups[0]['lr'])
        scheduler = lr_scheduler.ReduceLROnPlateau(trainer.optimizer, 'min', factor=config.pretraining_lr_factor, patience=config.pretraining_patience)
        best_val_loss = 0.
        for epoch in range(config.pretraining_num_epochs):
            print("========= Epoch %d of %d =========" % (epoch+1, config.pretraining_num_epochs))
            train_losses = trainer.train(train_dataset)
            valid_losses, valid_accuracy = trainer.test(valid_dataset)
            valid_loss = sum(valid_losses)
            scheduler.step(valid_loss)
            if (epoch==0) or (valid_loss<best_val_loss):
                print('Saving checkpoint')
                trainer.save_checkpoint()
                best_val_loss = valid_loss

            print("========= Results: epoch %d of %d =========" % (epoch+1, config.pretraining_num_epochs))
            print('Learning rate: %.10f' %trainer.optimizer.param_groups[0]['lr'])
            for i_loss, loss_label in enumerate(trainer.model.losses):
                print("%s: train loss: %.4f| valid loss: %.4f| valid accuracy: %.4f\n" % (loss_label, train_losses[i_loss], valid_losses[i_loss], valid_accuracy[i_loss]) )

    if test:

        trainer.load_checkpoint() # test with best model (acc. to validation)
        test_losses, test_accuracy = trainer.test(test_dataset, set='test')
        print("========= Pretrain results =========")
        for i_loss, loss_label in enumerate(trainer.model.losses):
            print("%s: test loss: %.4f\n" % (loss_label, test_losses[i_loss]) )
            print("%s: test accuracy: %.4f\n" % (loss_label, test_accuracy[i_loss]) )

    if train or train_test:
        trainer, train_dataset, valid_dataset, test_dataset = setup_training(config, datapath, num_workers, is_cpu)

    if train:

        print('Learning rate: %.10f' %trainer.optimizer.param_groups[0]['lr'])
        scheduler = lr_scheduler.ReduceLROnPlateau(trainer.optimizer, 'min', patience=config.training_patience)
        best_val_loss = 0.

        for epoch in range(config.training_num_epochs):
            print("========= Epoch %d of %d =========" % (epoch+1, config.training_num_epochs))
            train_losses = trainer.train(train_dataset)
            # train_losses = [0.,0.,0.]
            valid_losses, valid_accuracy = trainer.test(valid_dataset)
            valid_loss = sum(valid_losses)
            scheduler.step(valid_loss)
            if (epoch==0) or (valid_loss<best_val_loss):
                print('Saving checkpoint')
                trainer.save_checkpoint()
                best_val_loss = valid_loss

            print("========= Results: epoch %d of %d =========" % (epoch+1, config.pretraining_num_epochs))
            print('Learning rate: %.10f' %trainer.optimizer.param_groups[0]['lr'])
            for i_loss, loss_label in enumerate(trainer.model.losses):
                print("%s: train loss: %.4f| valid loss: %.4f| valid accuracy: %.4f\n" % (loss_label, train_losses[i_loss], valid_losses[i_loss], valid_accuracy[i_loss]) )

    if train_test:
        trainer.load_checkpoint()
        trainer.model.pretrained_model.eval()
        test_losses, test_accuracy, _ = trainer.test(test_dataset, set='test')
        print("========= Train results =========")
        for i_loss, loss_label in enumerate(trainer.model.losses):
            print("%s: test loss: %.4f\n" % (loss_label, test_losses[i_loss]) )
            print("%s: test accuracy: %.4f\n" % (loss_label, test_accuracy[i_loss]) )
