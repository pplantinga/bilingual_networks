"""An acoustic model that outputs both word and phone probabilities.

Authors
 * Peter Donhauser 2024
 * Peter Plantinga 2025
"""

import torch

import speechbrain as sb


class WordPhoneModel(torch.nn.Module):
    """This model uses convolutional and recurrent layers to model
    both word and phoneme probabilities. 

    This model expects 3-dimensional input [batch, time, feats] and
    by default produces three output with sizes:
     * [batch, time, word_outputs]
     * [batch, time, phone_outputs]
     * [batch, time, homolog_outputs]

    Arguments
    ---------
    languages : int
        Number of languages, for computing language embedding.
    word_outputs : int
        Size of the word output layer.
    phone_outputs : int
        Size of the phone output layer.
    homolog_outputs : int, optional
        Size of the homolog output layer, pass None to disable.
    input_size : int
        The length of the expected input at the third dimension.
    input_shape : tuple
        While input_size will suffice, this option can allow putting
        CRDNN into a sequential with other classes.
    activation : torch class
        A class used for constructing the activation layers for CNN and DNN.
    dropout : float
        Neuron dropout rate as applied to CNN, RNN, and DNN.
    cnn_blocks : int
        The number of convolutional neural blocks to include.
    cnn_channels : list of ints
        A list of the number of output channels for each CNN block.
    cnn_kernelsize : tuple of ints
        The size of the convolutional kernels.
    rnn_class : torch class
        The type of RNN to use in CRDNN network (LiGRU, LSTM, GRU, RNN)
    rnn_layers : int
        The number of recurrent RNN layers to include.
    rnn_neurons : int
        Number of neurons in each layer of the RNN.

    Example
    -------
    >>> inputs = torch.rand([10, 15, 60])
    >>> model = CRDNN(2, 100, 50, 20, input_shape=inputs.shape)
    >>> wrd_out, phn_out, hlg_out = model(inputs)
    >>> wrd_out.shape
    torch.Size([10, 15, 100])
    >>> phn_out.shape
    torch.Size([10, 15, 50])
    >>> hlg_out.shape
    torch.Size([10, 15, 20])
    """

    def __init__(
        self,
        languages,
        word_outputs,
        phone_outputs,
        homolog_outputs=None,
        input_size=None,
        input_shape=None,
        activation=torch.nn.LeakyReLU,
        cnn_blocks=3,
        cnn_channels=128,
        cnn_kernelsize=3,
        lang_embedding_size=5,
        chance_lang_unknown=0.5,
        rnn_class=sb.nnet.RNN.GRU,
        rnn_layers=1,
        rnn_neurons=256,
        rnn_dropout=0.2,
        rnn_bidirectional=False,
    ):
        super().__init__()
        if input_size is None and input_shape is None:
            raise ValueError("Must specify one of input_size or input_shape")
        if input_shape is None:
            input_shape = [None, None, input_size]

        self.CNN = sb.nnet.containers.Sequential(input_shape=input_shape)
        for block_index in range(cnn_blocks):
            self.CNN.append(
                CNN_Block,
                channels=cnn_channels,
                kernel_size=cnn_kernelsize,
                activation=activation,
                layer_name=f"block_{block_index}",
            )

        # Language embedding is concatenated to CNN output. Add one to language count
        # for "unknown language" tag, used with probability chance_lang_unknown
        self.chance_lang_unknown = chance_lang_unknown
        self.lang_embedding = sb.nnet.embedding.Embedding(
            num_embeddings=languages + 1,
            embedding_dim=lang_embedding_size,
        )

        # First level RNN takes CNN + language embedding as inputs
        self.phoneRNN = rnn_class(
            input_shape=[64, 100, cnn_channels + lang_embedding_size],
            hidden_size=rnn_neurons,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            bidirectional=rnn_bidirectional,
        )

        self.dropout = torch.nn.Dropout(rnn_dropout)

        # Create intermediate outputs
        self.phone_out = sb.nnet.linear.Linear(
            input_size=rnn_neurons, n_neurons=phone_outputs
        )
        self.homolog_outputs = homolog_outputs
        if homolog_outputs:
            self.homolog_out = sb.nnet.linear.Linear(
                input_size=rnn_neurons, n_neurons=homolog_outputs
            )

        # Second level of RNN
        self.wordRNN = rnn_class(
            input_shape=[64, 100, rnn_neurons],
            hidden_size=rnn_neurons,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            bidirectional=False,
        )

        # Final output
        self.word_out = sb.nnet.linear.Linear(
            input_size=rnn_neurons, n_neurons=word_outputs
        )

    def forward(self, mel_spectrogram, language):
        """Phone and Homolog outputs are after the first RNN, Words are after second RNN

        Arguments
        ---------
        mel_spectrogram: torch.FloatTensor
            Input signal tensor of shape [batch, time, n_mels]
        language: torch.LongTensor
            The language id (integer) of each sample, shape [batch]
        """

        # Produces [batch, time, cnn_out]
        cnn_out = self.CNN(mel_spectrogram)

        # Produces [batch, 1, lang_embed_size], some chance "unknown"
        language *= torch.rand_like(language.float()) < self.chance_lang_unknown
        lang_embedding = self.lang_embedding(language).unsqueeze(1)

        # Pass to RNN the CNN+lang_embedding concatenated on channel axis
        lang_embedding = lang_embedding.repeat(1, cnn_out.size(1), 1)
        rnn_in = torch.cat([cnn_out, lang_embedding], dim=2)
        phone_rnn_out = self.dropout(self.phoneRNN(rnn_in)[0])

        # Produce intermediate outputs [batch, time, phone_out]
        phone_out = self.phone_out(phone_rnn_out)

        # Last level RNN and output
        word_rnn_out = self.dropout(self.wordRNN(phone_rnn_out)[0])
        word_out = self.word_out(word_rnn_out)

        # If homologs requested, make 3 outputs
        if self.homolog_outputs:
            homolog_out = self.homolog_out(phone_rnn_out)
            return word_out, phone_out, homolog_out
        
        # Otherwise just return two levels
        return word_out, phone_out


class CNN_Block(sb.nnet.containers.Sequential):
    """CNN Block, based on VGG blocks.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input.
    channels : int
        Number of convolutional channels for the block.
    kernel_size : int
        Size of the 1d convolutional kernel
    activation : torch.nn.Module class
        A class to be used for instantiating an activation layer.
    dropout : float
        Rate to use for dropping channels.

    Example
    -------
    >>> inputs = torch.rand(10, 15, 60)
    >>> block = CNN_Block(input_shape=inputs.shape, channels=32)
    >>> outputs = block(inputs)
    >>> outputs.shape
    torch.Size([10, 15, 32])
    """

    def __init__(
        self,
        input_shape,
        channels,
        kernel_size=3,
        activation=torch.nn.LeakyReLU,
    ):
        super().__init__(input_shape=input_shape)
        self.append(
            sb.nnet.CNN.Conv1d,
            out_channels=channels,
            kernel_size=kernel_size,
            layer_name="conv",
        )
        self.append(sb.nnet.normalization.LayerNorm, layer_name="norm")
        self.append(activation(), layer_name="activation")
