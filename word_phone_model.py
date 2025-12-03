"""An acoustic model that outputs both word and phone probabilities.

Authors
 * Peter Donhauser 2024
 * Peter Plantinga 2025
"""

import torch
import torch.nn as nn


class WordPhoneModel(nn.Module):
    """This model uses convolutional and recurrent layers to model
    both word and phoneme probabilities. 

    This model expects 3-dimensional input [batch, time, feats] and
    produces three outputs with sizes:
     * [batch, time, phone_outputs]
     * [batch, time, lang_outputs]
     * [batch, time, word_outputs]

    Arguments
    ---------
    phone_outputs : int
        Size of the phoneme output layer.
    lang_outputs : int
        Number of languages in the lang detection output.
    word_outputs : int
        Size of the word output layer.
    input_size : int
        The length of the expected input at the third dimension.
    activation : torch.nn.Module class
        A class used for constructing the activation layers for CNN.
    cnn_blocks : int
        The number of convolutional neural blocks to include.
    cnn_channels : int
        Number of output channels for each CNN block.
    cnn_kernelsize : int
        The size of the convolutional kernels.
    rnn_class : torch.nn.Module class
        The type of RNN to use (LSTM, GRU, RNN).
    rnn_layers : int
        The number of recurrent RNN layers to include.
    rnn_neurons : int
        Number of neurons in each layer of the RNN.
    rnn_dropout : float
        Dropout rate for RNN and linear layers.
    rnn_bidirectional : bool
        Whether to use bidirectional RNN for phone layer.
    phone_bottleneck : bool
        Whether to insert the phone layer between RNNs.
    phone_softmax : bool
        Whether to add a softmax to force a single phoneme selection per step.
        Does nothing unless `phone_bottleneck` is also `True`.
    phone_softmax_temp : float
        Starting temperature for softmax (before decay), usually > 1.0
        Default 2.0
    phone_softmax_temp_step : float
        Decay rate for softmax, loses this much temp per step.
        Default 0.001
    phone_softmax_temp_min : float
        Ending value for softmax after decay is complete, usually < 1.0
        Default 0.2

    Example
    -------
    >>> inputs = torch.rand([10, 15, 60])
    >>> model = WordPhoneModel(
    ...   phone_outputs=50, lang_outputs=3, word_outputs=100, input_size=inputs.size(-1)
    ... )
    >>> phone_out, lang_out, word_out = model(inputs, torch.zeros(10, dtype=torch.long))
    >>> phone_out.shape
    torch.Size([10, 15, 50])
    >>> lang_out.shape
    torch.Size([10, 15, 3])
    >>> word_out.shape
    torch.Size([10, 15, 100])
    """

    def __init__(
        self,
        phone_outputs,
        lang_outputs,
        word_outputs,
        input_size,
        activation=nn.LeakyReLU,
        cnn_blocks=3,
        cnn_channels=128,
        cnn_kernelsize=3,
        lang_embedding_size=5,
        chance_lang_unknown=0.5,
        rnn_class=nn.GRU,
        rnn_layers=1,
        rnn_neurons=256,
        rnn_dropout=0.2,
        rnn_bidirectional=False,
        phone_bottleneck=False,
        phone_softmax=False,
        phone_softmax_temp=2.0,
        phone_softmax_temp_step=0.001,
        phone_softmax_temp_min=0.2,
    ):
        super().__init__()

        # Build CNN blocks
        cnn_layers = []
        in_channels = input_size
        for _ in range(cnn_blocks):
            cnn_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=cnn_channels,
                    kernel_size=cnn_kernelsize,
                    padding=cnn_kernelsize // 2,
                )
            )
            cnn_layers.append(nn.BatchNorm1d(cnn_channels))
            cnn_layers.append(activation())
            in_channels = cnn_channels
        self.cnn = nn.Sequential(*cnn_layers)

        # First level RNN (phone level)
        self.phone_rnn = rnn_class(
            input_size=cnn_channels,
            hidden_size=rnn_neurons,
            num_layers=rnn_layers,
            dropout=rnn_dropout if rnn_layers > 1 else 0,
            bidirectional=rnn_bidirectional,
            batch_first=True,
        )

        self.dropout = nn.Dropout(rnn_dropout)

        # Intermediate outputs
        self.phone_out = nn.Linear(rnn_neurons, phone_outputs)
        self.lang_out = nn.Linear(rnn_neurons, lang_outputs)

        # Settings for bottleneck
        self.phone_bottleneck = phone_bottleneck
        self.phone_softmax = phone_softmax
        self.temp = phone_softmax_temp
        self.temp_step = phone_softmax_temp_step
        self.temp_min = phone_softmax_temp_min
        
        # Second level RNN (word level)
        word_input = phone_outputs if phone_bottleneck else rnn_neurons
        self.word_rnn = rnn_class(
            input_size=word_input,
            hidden_size=rnn_neurons,
            num_layers=rnn_layers,
            dropout=rnn_dropout if rnn_layers > 1 else 0,
            bidirectional=rnn_bidirectional,
            batch_first=True,
        )

        # Final output
        self.word_out = nn.Linear(rnn_neurons, word_outputs)

    def step_bottleneck_temp(self):
        """Reduce temperature by one step, until min is reached"""
        self.temp = max(self.temp_min, self.temp - self.temp_step)

    def forward(self, mel_spectrogram):
        """Forward pass through the model.

        Arguments
        ---------
        mel_spectrogram: torch.FloatTensor
            Input signal tensor of shape [batch, time, n_mels]

        Returns
        -------
        lang_out : torch.FloatTensor
            Language predictions [batch, time, lang_outputs]
        phone_out : torch.FloatTensor
            Phone predictions [batch, time, phone_outputs]
        word_out : torch.FloatTensor
            Word predictions [batch, time, word_outputs]
        """
        batch_size, time_steps, _ = mel_spectrogram.shape

        # CNN expects [batch, channels, time]
        cnn_out = self.cnn(mel_spectrogram.transpose(1, 2))
        # Back to [batch, time, channels]
        cnn_out = cnn_out.transpose(1, 2)

        # Phone-level RNN
        phone_rnn_out, _ = self.phone_rnn(cnn_out)
        phone_rnn_out = self.dropout(phone_rnn_out)

        # Language predictions
        lang_out = self.lang_out(phone_rnn_out)

        # Phone predictions
        phone_out = self.phone_out(phone_rnn_out)

        # Prepare bottleneck inputs to word rnn
        if self.phone_bottleneck:
            phone_rnn_out = phone_out
            if self.phone_softmax:
                phone_rnn_out = torch.softmax(phone_rnn_out / self.temp, dim=-1)

        # Word-level RNN
        word_rnn_out, _ = self.word_rnn(phone_rnn_out)
        word_rnn_out = self.dropout(word_rnn_out)
        
        # Word predictions
        word_out = self.word_out(word_rnn_out)

        return phone_out, lang_out, word_out
