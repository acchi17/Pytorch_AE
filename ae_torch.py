# Ref. of code
# https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self

        , image_channels
        , encoder_conv_filters
        , encoder_conv_kernel_size
        , encoder_conv_strides
        , decoder_conv_t_filters
        , decoder_conv_t_kernel_size
        , decoder_conv_t_strides
        , z_dim
        , use_batch_norm = False
        , use_dropout = False
        ):
        """
        Create an Autoencoding model.
        :param input:
        :return: -
        <Memo>
        - Outuput size of ConvTranspose2d(Vertical or Horizontal size):
          Output size=(Stride*(InputSize-1)+KernelSize-2*Padding+OutputPadding)
        """
        super().__init__()
        self.name = 'CustomAutoencoder'
        self.conv_out_size = None
        self.fc_in_size = None
        self.z_dim = z_dim
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        # Build Encoder
        Layers = []
        in_channels = image_channels
        for i in range(len(encoder_conv_filters)):
            Layers.append(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=encoder_conv_filters[i],
                          kernel_size=encoder_conv_kernel_size[i],
                          stride=encoder_conv_strides[i],
                          padding=1)
            )
            if self.use_batch_norm:
                Layers.append(nn.BatchNorm2d(encoder_conv_filters[i]))
            # ReLU is more general and effective than LeakyReLU
            #Layers.append(nn.LeakyReLU())
            Layers.append(nn.ReLU(inplace=True))  
            if self.use_dropout:
                Layers.append(nn.Dropout(p=0.3))
            in_channels = encoder_conv_filters[i]

        self.encoder_cnn = nn.Sequential(*Layers)
        self.encoder_out = None

        # Build Decoder
        #self.decoder_in = nn.Linear(self.z_dim, 3136)
        self.decoder_in = None
        Layers = []
        in_channels = encoder_conv_filters[-1]
        for i in range(len(decoder_conv_t_filters)):
            Layers.append(
                nn.ConvTranspose2d(in_channels=in_channels,
                                   out_channels=decoder_conv_t_filters[i],
                                   kernel_size=decoder_conv_t_kernel_size[i],
                                   stride=decoder_conv_t_strides[i],
                                   padding=1,
                                   output_padding=decoder_conv_t_strides[i]-1),

            )
            if self.use_batch_norm and (i < len(decoder_conv_t_filters)-1):
                Layers.append(nn.BatchNorm2d(decoder_conv_t_filters[i]))
            if (i < len(decoder_conv_t_filters)-1):
                # ReLU is more general and effective than LeakyReLU
                #Layers.append(nn.LeakyReLU())
                Layers.append(nn.ReLU(inplace=True))
            else:
                # The output values of the final layer are set to be
                # within the range of values of the training data
                # and tanh reduces vanishing gradient more than sigmoid
                #Layers.append(nn.Sigmoid())
                Layers.append(nn.Tanh())
            if self.use_dropout and (i < len(decoder_conv_t_filters)-1):
                Layers.append(nn.Dropout(p=0.3))
            in_channels = decoder_conv_t_filters[i]
            
        self.decoder_cnn = nn.Sequential(*Layers)

    def encode(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        x = self.encoder_cnn(x)
        self.conv_out_size = list(x.size())
        x = torch.flatten(x, start_dim=1)
        if self.encoder_out is None:
            self.fc_in_size = x.size(1)  # x.size(): ex. torch.Size([32, 3136])
            self.encoder_out = nn.Linear(self.fc_in_size, self.z_dim)
            # Move to the same device as input tensor created dynamically
            self.encoder_out = self.encoder_out.to(x.device)
            x = self.encoder_out(x)               
        return x

    def decode(self, x):
        """
        Decodes the latent code z by passing through the decoder network
        and returns the reconstructed image.
        :param z: (Tensor) Latent code [N x z_dim]
        :return: (Tensor) Reconstructed image [N x C x H x W]
        """
        if self.decoder_in is None:
            self.decoder_in = nn.Linear(self.z_dim, self.fc_in_size)
            # Move to the same device as input tensor
            self.decoder_in = self.decoder_in.to(x.device)
            x = self.decoder_in(x)
        self.conv_out_size[0] = -1
        x = x.contiguous().view(self.conv_out_size)
        x = self.decoder_cnn(x)
        return x


    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
