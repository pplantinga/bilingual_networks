"""An acoustic model that outputs both word and phone probabilities.

Rewritten to use only PyTorch primitives (no SpeechBrain dependency).

Authors
 * Peter Donhauser 2024
 * Peter Plantinga 2025
 * Rewritten with PyTorch primitives 2025
"""

import torch
import torch.nn as nn


class WordPhoneModel(nn.Module):
    """This model uses convolutional and recurrent layers to model
    both word and phoneme probabilities. 

    This model expects 3-dimensional input [batch, time, feats] and
    produces two outputs with sizes:
     * [batch, time, word_outputs]
     * [batch, time, phone_outputs]

    Plus an optional third output with size:
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
    activation : torch.nn.Module class
        A class used for constructing the activation layers for CNN.
    cnn_blocks : int
        The number of convolutional neural blocks to include.
    cnn_channels : int
        Number of output channels for each CNN block.
    cnn_kernelsize : int
        The size of the convolutional kernels.
    lang_embedding_size : int
        Dimension of language embeddings.
    chance_lang_unknown : float
        Probability of masking language during training.
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

    Example
    -------
    >>> inputs = torch.rand([10, 15, 60])
    >>> model = WordPhoneModel(2, 100, 50, 20, input_size=60)
    >>> wrd_out, phn_out, hlg_out = model(inputs, torch.zeros(10, dtype=torch.long))
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
    ):
        super().__init__()
        if input_size is None:
            raise ValueError("Must specify input_size")

        self.chance_lang_unknown = chance_lang_unknown
        self.homolog_outputs = homolog_outputs

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

        # Language embedding (+1 for "unknown language")
        self.lang_embedding = nn.Embedding(
            num_embeddings=languages + 1,
            embedding_dim=lang_embedding_size,
        )

        # First level RNN (phone level)
        self.phone_rnn = rnn_class(
            input_size=cnn_channels + lang_embedding_size,
            hidden_size=rnn_neurons,
            num_layers=rnn_layers,
            dropout=rnn_dropout if rnn_layers > 1 else 0,
            bidirectional=rnn_bidirectional,
            batch_first=True,
        )

        self.dropout = nn.Dropout(rnn_dropout)

        # Intermediate outputs
        self.phone_out = nn.Linear(rnn_neurons, phone_outputs)
        
        if homolog_outputs:
            self.homolog_out = nn.Linear(rnn_neurons, homolog_outputs)

        # Second level RNN (word level)
        self.word_rnn = rnn_class(
            input_size=rnn_neurons,
            hidden_size=rnn_neurons,
            num_layers=rnn_layers,
            dropout=rnn_dropout if rnn_layers > 1 else 0,
            bidirectional=rnn_bidirectional,
            batch_first=True,
        )

        # Final output
        self.word_out = nn.Linear(rnn_neurons, word_outputs)

    def forward(self, mel_spectrogram, language):
        """Forward pass through the model.

        Arguments
        ---------
        mel_spectrogram: torch.FloatTensor
            Input signal tensor of shape [batch, time, n_mels]
        language: torch.LongTensor
            The language id (integer) of each sample, shape [batch]

        Returns
        -------
        word_out : torch.FloatTensor
            Word predictions [batch, time, word_outputs]
        phone_out : torch.FloatTensor
            Phone predictions [batch, time, phone_outputs]
        homolog_out : torch.FloatTensor (optional)
            Homolog predictions [batch, time, homolog_outputs]
        """
        batch_size, time_steps, _ = mel_spectrogram.shape

        # CNN expects [batch, channels, time]
        x = mel_spectrogram.transpose(1, 2)
        cnn_out = self.cnn(x)
        # Back to [batch, time, channels]
        cnn_out = cnn_out.transpose(1, 2)

        # Language embedding with random masking during training
        if self.training:
            mask = torch.rand(batch_size, device=language.device) < self.chance_lang_unknown
            language_masked = language * (~mask).long()
        else:
            language_masked = language
        
        # [batch, lang_embedding_size]
        lang_emb = self.lang_embedding(language_masked)
        # Expand to [batch, time, lang_embedding_size]
        lang_emb = lang_emb.unsqueeze(1).expand(-1, time_steps, -1)

        # Concatenate CNN output with language embedding
        rnn_in = torch.cat([cnn_out, lang_emb], dim=2)
        
        # Phone-level RNN
        phone_rnn_out, _ = self.phone_rnn(rnn_in)
        phone_rnn_out = self.dropout(phone_rnn_out)

        # Phone predictions
        phone_out = self.phone_out(phone_rnn_out)

        # Word-level RNN
        word_rnn_out, _ = self.word_rnn(phone_rnn_out)
        word_rnn_out = self.dropout(word_rnn_out)
        
        # Word predictions
        word_out = self.word_out(word_rnn_out)

        # Optional homolog output
        if self.homolog_outputs:
            homolog_out = self.homolog_out(phone_rnn_out)
            return word_out, phone_out, homolog_out
        
        return word_out, phone_out
