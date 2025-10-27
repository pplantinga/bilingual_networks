import torch
import torch.utils.data
import configparser
import os
from glob import glob
import multiprocessing
from subprocess import call
import soundfile as sf
import numpy as np
import textgrid
from collections import Counter
import json
from tqdm import tqdm
import pandas as pd

import torch
import torch.utils.data

from params import lang_order


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        # dataset_type = type(dataset)
        # if dataset_type is torchvision.datasets.MNIST:
        #     return dataset.train_labels[idx].item()
        # elif dataset_type is torchvision.datasets.ImageFolder:
        #     return dataset.imgs[idx][1]
        # else:
        #     raise NotImplementedError
        return dataset.lang_ind[idx]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, wav_paths, config):
        """
        wav_paths: list of strings (wav file paths)
        config: Config object (contains info about model and training)
        """
        self.wav_paths = wav_paths # list of wav file paths
        self.length_mean = config.pretraining_length_mean
        self.length_var = config.pretraining_length_var

        self.loader = torch.utils.data.DataLoader(self, batch_size=config.pretraining_batch_size, shuffle=True, collate_fn=CollateWavs())

    def __len__(self):
        return len(self.wav_paths)

    def read_audio(self, idx):
        x, fs = sf.read(self.wav_paths[idx])
        if x.ndim>1:
            return x.sum(1), fs
        else:
            return x, fs

    def __getitem__(self, idx):
        x, fs = self.read_audio(idx)

        # # Cut a snippet of length random_length from the audio
        # random_length = round(fs * max(self.length_mean + self.length_var * torch.randn(1).item(), 0.5))
        # if len(x) <= random_length:
        #     start = 0
        # else:
        #     start = torch.randint(low=0, high=len(x)-random_length, size=(1,)).item()
        # end = start + random_length
        #
        # x = x[start:end]

        return x, self.wav_paths[idx]

class ASRDataset(AudioDataset):
    def __init__(self, wav_paths, lang_ind, textgrid_paths, Sy_phoneme, Sy_word, Sy_homologe, config, train=True, num_samples=None, num_workers=10):
        """
        wav_paths: list of strings (wav file paths)
        textgrid_paths: list of strings (textgrid for each wav file)
        lang_ind: list of integers (language index for each wav file)
        Sy_phoneme: list of strings (all possible phonemes)
        Sy_word: list of strings (all possible words)
        config: Config object (contains info about model and training)
        """
        super(ASRDataset,self).__init__(wav_paths, config)

        self.textgrid_paths = textgrid_paths # list of textgrid file paths

        languages = np.array(lang_order)
        self.languages = languages
        self.lang_ind = np.zeros(len(lang_ind), dtype=int)
        for i_lang, language in enumerate(languages):
            self.lang_ind[lang_ind==language] = i_lang
        self.lang_sil = 0
        # self.Sy_phoneme = np.array([np.array(Sy) for Sy in Sy_phoneme])
        self.Sy_phoneme = Sy_phoneme
        self.phonemes = np.concatenate(Sy_phoneme)
        self.Sy_homologe = Sy_homologe
        self.ind_nospeech = np.where((self.phonemes=='sil')|(self.phonemes=='sp')|(self.phonemes=='spn'))[0]
        if len(self.ind_nospeech)==0:
            self.ind_nospeech = np.array([-1,-1,-1])
        self.num_phonemes = sum([len(phones) for phones in Sy_phoneme])
        # self.Sy_word = np.array(Sy_word)
        self.Sy_word = Sy_word
        self.words = np.concatenate(Sy_word)
        self.num_words = sum([len(words) for words in Sy_word])
        self.downsample_factor = config.downsample_factor
        speakers = np.array([wav_path.split('/')[-1][:10] for wav_path in wav_paths])
        self.speakers = np.zeros(len(speakers),dtype=int)
        if train:
            self.loader = torch.utils.data.DataLoader(self, batch_size=config.pretraining_batch_size, pin_memory=True,
                                                      num_workers=num_workers, shuffle=False, collate_fn=CollateItems(), # shuffle off because using sampler
                                                      sampler=ImbalancedDatasetSampler(self, num_samples=num_samples))
        else:
            for i_speaker, speaker in enumerate(tqdm(np.unique(speakers))):
                self.speakers[speakers==speaker] = i_speaker
            self.loader = torch.utils.data.DataLoader(self, batch_size=8, pin_memory=True,
                                                      num_workers=num_workers, shuffle=False, collate_fn=CollateItems())

    def __getitem__(self, idx, snippet=None):
        x, fs = self.read_audio(idx)

        # Downsample from 48000 to 16000
        assert fs == 48000
        fs = 16000
        x = x[::3]

        y_lang = self.lang_ind[idx]

        if os.path.isfile(self.textgrid_paths[idx]):

            tg = textgrid.TextGrid()
            tg.read(self.textgrid_paths[idx])

            y_phoneme, y_homologe = [], []
            for phoneme in tg.getList("phones")[0]:
                duration = phoneme.maxTime - phoneme.minTime
                phoneme = phoneme.mark.rstrip("0123456789")
                if phoneme in ['sil', 'sp', 'spn']:
                    index_offset = sum([len(ph) for ph in self.Sy_phoneme[:self.lang_sil]])
                    phoneme_index = self.Sy_phoneme[self.lang_sil].index(phoneme) + index_offset
                    homologe_index = self.Sy_homologe[self.lang_sil][phoneme]
                else:
                    if phoneme in self.Sy_phoneme[y_lang]:
                        index_offset = sum([len(ph) for ph in self.Sy_phoneme[:y_lang]])
                        phoneme_index = self.Sy_phoneme[y_lang].index(phoneme) + index_offset
                        homologe_index = self.Sy_homologe[y_lang][phoneme]
                    else:
                        phoneme_index = -1
                        homologe_index = -1
                if phoneme == '': phoneme_index = -1
                y_phoneme += [phoneme_index] * round(duration * fs)
                y_homologe += [homologe_index] * round(duration * fs)

            y_word = []
            y_iword = []
            for i_word, word in enumerate(tg.getList("words")[0]):
                duration = word.maxTime - word.minTime
                index_offset = sum([len(ph) for ph in self.Sy_word[:y_lang]])
                word_index = self.Sy_word[y_lang].index(word.mark)+index_offset if word.mark in self.Sy_word[y_lang] else -1
                if word.mark == '': word_index = -1
                y_word += [word_index] * round(duration * fs)
                y_iword += [i_word] * round(duration * fs) # word count cumulative

        else:

            y_phoneme = np.ones(len(x)) * -1
            y_word = np.ones(len(x)) * -1
            y_iword = np.zeros(len(x))

        # Cut a snippet of length random_length from the audio
        if snippet is None:
            if self.length_mean>.5:
                random_length = round(fs * max(self.length_mean + self.length_var * torch.randn(1).item(), 0.5))
            else:
                random_length = round(fs * max(self.length_mean + self.length_var * torch.randn(1).item(), 0.05))
            if len(x) <= random_length:
                start = 0
                end = len(x)
            else:
                start = torch.randint(low=0, high=len(x)-random_length, size=(1,)).item()
                end = start + random_length
        else:
            start = snippet[0]
            end = snippet[1]

        # (I convert everything to numpy arrays, since there's a memory leak otherwise)
        x = x[start:end]
        # x[start_noise:end_noise] = np.random.randn(noise_len)/10
        y_phoneme = np.array(y_phoneme[start:end:self.downsample_factor])
        y_homologe = np.array(y_homologe[start:end:self.downsample_factor])
        y_word = np.array(y_word[start:end:self.downsample_factor])
        y_lang = y_lang.repeat(len(y_phoneme))
        y_speech = np.ones(y_lang.shape)
        y_speech[(self.ind_nospeech[0]==y_phoneme)|(self.ind_nospeech[1]==y_phoneme)|(self.ind_nospeech[2]==y_phoneme)] = 0
        # y_speaker = self.speakers[idx].repeat(len(y_phoneme))
        y_speaker = np.array([idx]).repeat(len(y_phoneme))
        y_iword = np.array(y_iword[start:end:self.downsample_factor])
        return (x, y_lang, y_phoneme, y_word, y_homologe, y_speech, y_speaker, y_iword)

class CollateWavs:
    def __call__(self, batch):
        """
        batch: list of tuples (input wav, phoneme labels, word labels)

        Returns a minibatch of wavs and labels as Tensors.
        """
        x = []; y = []; lengths = []
        batch_size = len(batch)
        for index in range(batch_size):
            x_, y_ = batch[index]

            x.append(torch.tensor(x_).float())
            # placeholder labels:
            y.append(y_)
            lengths.append(torch.tensor(x_.shape[0]).long())

        # pad all sequences to have same length
        x = torch.nn.utils.rnn.pad_sequence(x).T
        # y = torch.nn.utils.rnn.pad_sequence(y).T
        lengths = torch.stack(lengths)

        return (x, y, lengths)

class CollateItems:
    def __call__(self, batch):
        """
        batch: list of tuples (input wav, phoneme labels, word labels)
        Returns a minibatch of wavs and labels as Tensors.
        """

        n_items = len(batch[0])
        batch_size = len(batch)
        x = []
        ys = [[] for ii in range(n_items-1)]
        lengths = []
        for index in range(batch_size):
            items_ = batch[index]
            x.append(torch.tensor(items_[0]).float())
            for item_, y in zip(items_[1:], ys):
                y.append( torch.tensor(item_).long() )
            lengths.append(torch.tensor(items_[0].shape[0]).long())
        x = torch.nn.utils.rnn.pad_sequence(x).T
        for i_y in range(len(ys)):
            ys[i_y] = torch.nn.utils.rnn.pad_sequence(ys[i_y]).T
        lengths = torch.stack(lengths)

        return (x, tuple(ys), lengths)

class Config:
    def __init__(self):
        self.thats_me = True

def read_config(config_file):
    config = Config()
    parser = configparser.ConfigParser()
    parser.read(config_file)

    #[experiment]
    config.seed=int(parser.get("experiment", "seed"))
    config.folder = os.path.splitext(config_file)[0]
    # config.folder=parser.get("experiment", "folder")

    # Make a folder containing experiment information
    if not os.path.isdir(config.folder):
        os.mkdir(config.folder)
        os.mkdir(os.path.join(config.folder, "pretraining"))
        os.mkdir(os.path.join(config.folder, "training"))
    call("cp " + config_file + " " + os.path.join(config.folder, "experiment.cfg"), shell=True)

    #[model]
    config.type=parser.get("model", "type")
    config.n_mel=int(parser.get("model", "n_mel"))
    config.n_mfcc_out=int(parser.get("model", "n_mfcc_out"))
    config.cnn_out_size=int(parser.get("model", "cnn_out_size"))
    config.rnn_hidden_size=int(parser.get("model", "rnn_hidden_size"))
    config.num_rnn_layers=int(parser.get("model", "num_rnn_layers"))
    config.rnn_drop=float(parser.get("model", "rnn_drop"))
    config.rnn_bidirectional=parser.get("model", "rnn_bidirectional") == "True"
    config.downsample_factor=int(parser.get("model", "downsample_factor"))
    config.vocabulary_size=int(parser.get("model", "vocabulary_size"))

    #[pretraining]
    config.pretraining_manifest_train=parser.get("pretraining", "pretraining_manifest_train")
    config.pretraining_manifest_dev=parser.get("pretraining", "pretraining_manifest_dev")
    config.pretraining_manifest_test=parser.get("pretraining", "pretraining_manifest_test")
    config.fs=int(parser.get("pretraining", "fs"))
    config.time_shift=int(parser.get("pretraining", "time_shift"))
    config.n_output_quantize=int(parser.get("pretraining", "n_output_quantize"))
    config.pretraining_lr=float(parser.get("pretraining", "pretraining_lr"))
    config.pretraining_patience=float(parser.get("pretraining", "pretraining_patience"))
    config.pretraining_lr_factor=float(parser.get("pretraining", "pretraining_lr_factor"))
    config.pretraining_batch_size=int(parser.get("pretraining", "pretraining_batch_size"))
    config.pretraining_num_epochs=int(parser.get("pretraining", "pretraining_num_epochs"))
    config.pretraining_length_mean=float(parser.get("pretraining", "pretraining_length_mean"))
    config.pretraining_length_var=float(parser.get("pretraining", "pretraining_length_var"))
    config.grad_clip=float(parser.get("pretraining", "grad_clip"))
    config.pretraining_phoneout=int(parser.get("pretraining", "pretraining_phoneout"))
    config.pretraining_wordout=int(parser.get("pretraining", "pretraining_wordout"))
    config.pretraining_langin=int(parser.get("pretraining", "pretraining_langin"))
    config.pretraining_langin_test=int(parser.get("pretraining", "pretraining_langin_test"))

    #[training]
    config.training_manifest_train=parser.get("training", "training_manifest_train")
    config.training_manifest_dev=parser.get("training", "training_manifest_dev")
    config.training_manifest_test=parser.get("training", "training_manifest_test")
    config.training_type=parser.get("training", "training_type")
    config.training_lr=float(parser.get("training", "training_lr"))
    config.training_patience=int(parser.get("training", "training_patience"))
    config.training_batch_size=int(parser.get("training", "training_batch_size"))
    config.training_num_epochs=int(parser.get("training", "training_num_epochs"))
    return config

def load_manifest(manifest_path, datapath):
    with open(os.path.join(datapath, manifest_path)) as f:
        wavfiles, tgfiles, languages = ([], [], [])
        for line in f.readlines():
            filename = line.strip('\n')
            languages.append(filename.split('/')[0])
            filepath = os.path.join(datapath, filename)
            wavfiles.append(filepath+'.mp3')
            tgfiles.append(filepath+'.TextGrid')
    return np.array(wavfiles), np.array(tgfiles), np.array(languages)

def get_datasets(config, datapath, manifest_train, manifest_dev, manifest_test, num_workers, folder='pretraining'):
    wavfiles, tgfiles, languages = load_manifest(manifest_train, datapath)
    tgfiles_vocab = tgfiles
    languages_vocab = languages
    Sy_phoneme, Sy_word, Sy_homologe = ([], [], [])
    for i_lang, language in enumerate(lang_order):
        df = pd.read_csv(f'data/phonemes_{language}.csv',index_col=0,na_filter=False)
        #Sy_phoneme.append(df.index.values.tolist())
        Sy_phoneme.append(df.ipa.values.tolist())
        Sy_homologe.append(df['homolog'].values.tolist())
        df = pd.read_csv(f'data/words_{language}.csv',index_col=0,na_filter=False)
        Sy_word.append(df['word'].values.tolist())
    homologes = np.unique(np.concatenate(Sy_homologe)).tolist()
    for i_lang in range(len(lang_order)):
        Sy_homologe[i_lang] = {ph: homologes.index(hom) for hom, ph in zip(Sy_homologe[i_lang],Sy_phoneme[i_lang])}
    config.num_phonemes = sum([len(phones) for phones in Sy_phoneme])
    config.num_homologes = len(homologes)
    config.vocabulary_size = sum([len(words) for words in Sy_word])
    config.languages = np.unique(languages_vocab)
    # Prepare training dataset
    wavfiles, tgfiles, languages = load_manifest(manifest_train, datapath)
    num_samples = len(wavfiles)
    train_dataset = ASRDataset(wavfiles, languages, tgfiles, Sy_phoneme, Sy_word, Sy_homologe,
                               config, True, num_samples, num_workers)
    pretraining_length_mean = config.pretraining_length_mean
    pretraining_length_var = config.pretraining_length_var
    pretraining_batch_size = config.pretraining_batch_size
    config.pretraining_length_mean = 100 # Use all
    config.pretraining_length_var = 0
    config.pretraining_batch_size = 4
    # Prepare validation and test datasets
    wavfiles, tgfiles, languages = load_manifest(manifest_dev, datapath)
    valid_dataset = ASRDataset(wavfiles, languages, tgfiles, Sy_phoneme, Sy_word, Sy_homologe,
                               config, False, num_workers=num_workers)
    wavfiles, tgfiles, languages = load_manifest(manifest_test, datapath)
    test_dataset = ASRDataset(wavfiles, languages, tgfiles, Sy_phoneme, Sy_word, Sy_homologe,
                              config, False, num_workers=num_workers)
    config.pretraining_length_mean = pretraining_length_mean
    config.pretraining_length_var = pretraining_length_var
    config.pretraining_batch_size = pretraining_batch_size

    return train_dataset, valid_dataset, test_dataset

def get_generic_dataset(config, path):
    wavfiles = glob(os.path.join(path, '*.wav'))
    dataset = AudioDataset(wavfiles, config)
    return [], [], dataset
