import torch
import torch.nn.functional as F
import torchaudio_contrib as tac
import os, math

class NCL2NLC(torch.nn.Module):
    def __init__(self):
        super(NCL2NLC, self).__init__()

    def forward(self, input):
        """
        input : Tensor of shape (batch size, T, Cin)

        Outputs a Tensor of shape (batch size, Cin, T).
        """

        return input.transpose(1,2)

class AmplitudeToDb(torch.nn.Module):
    def __init__(self, ref=1.0, amin=1e-7):
        super(AmplitudeToDb, self).__init__()
        self.ref = ref
        self.amin = amin

    def forward(self, input):
        input = torch.clamp(input, min=self.amin)
        return 10.0 * (torch.log10(input) - torch.log10(torch.tensor(self.ref, device=input.device, requires_grad=False)))

class SelectMels(torch.nn.Module):
    def __init__(self, n_mel_select):
        super(SelectMels, self).__init__()
        self.n_mel_select = n_mel_select

    def forward(self, input):
        return input[:,:self.n_mel_select,:]

def compute_criterion(inputs, targets, lengths, num_classes=(1, 2, 3)):
        losses = []
        accuracies = []
        for ii in range(len(num_classes)):
            inp_padded = torch.nn.utils.rnn.pack_padded_sequence(inputs[ii], lengths.cpu(), batch_first=True, enforce_sorted=False)
            tar_padded = torch.nn.utils.rnn.pack_padded_sequence(targets[ii], lengths.cpu(), batch_first=True, enforce_sorted=False)
            loss = F.cross_entropy(inp_padded.data, tar_padded.data, ignore_index=-1)
            valid_indices = (tar_padded.data!=-1)# & (tar_padded.data!=0)
            accuracy = (inp_padded.data.max(1)[1][valid_indices] == tar_padded.data[valid_indices]).float().mean()
            losses.append(loss)
            accuracies.append(accuracy)
        return sum(losses), losses, accuracies

class SpecNet(torch.nn.Module):

    def __init__(self, config):
        super(SpecNet, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        mel = tac.layers.Melspectrogram(128, config.fs, fft_length=config.n_fft,
        								hop_length=config.downsample_factor)
        self.feat = torch.nn.Sequential(mel, AmplitudeToDb(), SelectMels(config.n_mel))
        self.convnet = torch.nn.Sequential(
            torch.nn.Conv1d(config.n_mel, 64, kernel_size=(1), stride=1,
                                   bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(64, 128, kernel_size=(3), stride=1, padding=1,
                                   bias=False),
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(128, config.cnn_out_size, kernel_size=(3), stride=1, padding=1,
                                   bias=False),
            torch.nn.BatchNorm1d(config.cnn_out_size),
            torch.nn.LeakyReLU(),
            )
        self.ncl2nlc = NCL2NLC()
        self.drop = torch.nn.Dropout(config.rnn_drop)
        self.downsample_factor = config.downsample_factor
        self.cnn_out_size = config.cnn_out_size
        self.n_mel = config.n_mel
        if self.is_cuda:
            self.cuda()

    def compute_rnn_input(self, data):
        out = self.convnet(data[0])
        language = self.language(data[1])[:,:,None].repeat(1,1,data[0].shape[2])
        return self.ncl2nlc(torch.cat([out, language], 1))

    def features_to_stft(self, feat):
        # db to power
        amptodb = self.feat[1]
        feat = torch.pow(10.0, feat / 10.0 + torch.log10(torch.tensor(amptodb.ref,
                                                                device=feat.device,
                                                                requires_grad=False,
                                                                dtype=feat.dtype)))
        # filterbank inverse
        fbank = self.feat[0][2].filterbank
        feat = torch.matmul(feat.transpose(1,2), fbank.T)
        return feat

class ASRSpecNet(SpecNet):

    def __init__(self, config):
        super(ASRSpecNet, self).__init__(config)
        if config.num_rnn_layers==1:
            rnn_drop = 0.
        else:
            rnn_drop = config.rnn_drop
        n_languages = 3
        self.rnn_phone = torch.nn.GRU(input_size=config.cnn_out_size+(n_languages+1), hidden_size=config.rnn_hidden_size,
                                      num_layers=config.num_rnn_layers, dropout=rnn_drop,
                                      bidirectional=config.rnn_bidirectional, batch_first=True)
        self.out_dim = config.rnn_hidden_size
        if config.rnn_bidirectional:
            self.out_dim *= 2
        self.phone_linear = torch.nn.Linear(self.out_dim, config.num_phonemes+1)
        self.homologe_linear = torch.nn.Linear(self.out_dim, config.num_homologes+1)
        self.rnn_word = torch.nn.GRU(input_size=self.out_dim, hidden_size=config.rnn_hidden_size,
                                      num_layers=config.num_rnn_layers, dropout=rnn_drop,
                                      bidirectional=config.rnn_bidirectional, batch_first=True)
        self.word_linear = torch.nn.Linear(self.out_dim, config.vocabulary_size+1)
        # self.lang_linear = torch.nn.Linear(self.out_dim*2, 2)
        self.losses = ['phonemes', 'words', 'homologes']
        self.num_classes = [config.num_phonemes+1, config.vocabulary_size+1, config.num_homologes+1]
        self.language = torch.nn.Embedding(n_languages+1, n_languages+1)
        self.language.weight = torch.nn.Parameter(torch.eye(n_languages+1), requires_grad=False)
        if self.is_cuda:
            self.cuda()

    def forward(self, data, lengths, h0=None):
        cnn_out = self.compute_rnn_input(data)
        if h0 is not None:
            rnn_phone_output = self.drop(self.rnn_phone(cnn_out,
                h0[0].repeat(1,cnn_out.shape[0],1))[0])
        else:
            rnn_phone_output = self.drop(self.rnn_phone(cnn_out)[0])
        phone_out = self.phone_linear(rnn_phone_output)
        homologe_out = self.homologe_linear(rnn_phone_output)
        if h0 is not None:
            rnn_word_output = self.drop(self.rnn_word(rnn_phone_output,
                h0[1].repeat(1,cnn_out.shape[0],1))[0])
        else:
            rnn_word_output = self.drop(self.rnn_word(rnn_phone_output)[0])
        word_out = self.word_linear(rnn_word_output)
        return phone_out, word_out, homologe_out, cnn_out, rnn_phone_output, rnn_word_output

    def criterion(self, inputs, targets, lengths):
        return compute_criterion(inputs, targets, lengths, self.num_classes)


class ASRSpecNetPhonemes(torch.nn.Module):

    def __init__(self, config):
        super(ASRSpecNetPhonemes, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        pretrained_model = self.load_pretrained_model(config)
        self.pretrained_model = pretrained_model

        # Freeze pretrained model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.pretrained_model.eval()
        self.feat = self.pretrained_model.feat
        self.downsample_factor = self.pretrained_model.downsample_factor
        self.softmax = torch.nn.Softmax(dim=1)
        self.setup_last_layers(config)
        if self.is_cuda:
            self.cuda()

    def setup_last_layers(self, config):
        self.phone_linear = torch.nn.Linear(self.pretrained_model.out_dim, config.num_phonemes+1)
        self.losses = ['phonemes']
        self.num_classes = [config.num_phonemes+1]

    def load_pretrained_model(self, config):
        pretrained_model = ASRSpecNet(config)
        state_file = "model_state.pth"
        pretrained_model_path = os.path.join(config.folder, "pretraining", state_file)
        try:
            if self.is_cuda:
                pretrained_model.load_state_dict(torch.load(pretrained_model_path))
            else:
                pretrained_model.load_state_dict(torch.load(pretrained_model_path, map_location="cpu"))
        except Exception as e:
            print(e)
            # print('Could not load previous model.')
        return pretrained_model

    def get_pretrained_outputs(self, data, h0=None):
        _, _, _, _, rnn_phone_output, _ = self.pretrained_model.forward(data, None, h0)
        return rnn_phone_output

    def forward(self, data, lengths, h0=None):
        rnn_phone_output = self.get_pretrained_outputs(data, h0=h0)
        phone_out = self.phone_linear(rnn_phone_output)
        return phone_out

    def criterion(self, inputs, targets, lengths):
        return compute_criterion(inputs, targets, lengths, self.num_classes)
