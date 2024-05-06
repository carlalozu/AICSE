import torch
from torch import nn
from functions import CNOBlock, LiftProjectBlock, ResNet


class CNO1d(nn.Module):
    """CNO"""

    def __init__(self,
                 in_dim,                    # Number of input channels.
                 out_dim,                   # Number of input channels.
                 # Input and Output spatial size (required )
                 size,
                 # Number of (D) or (U) blocks in the network
                 N_layers,
                 # Number of (R) blocks per level (except the neck)
                 N_res=4,
                 N_res_neck=4,            # Number of (R) blocks in the neck
                 channel_multiplier=16,   # How the number of channels evolve?
                 use_bn=True,             # Add BN? We do not add BN in lifting/projection layer
                 ):

        super().__init__()

        self.N_layers = int(N_layers)         # Number od (D) & (U) Blocks
        # Input is lifted to the half of channel_multiplier dimension
        self.lift_dim = channel_multiplier//2
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.channel_multiplier = channel_multiplier  # The growth of the channels

        ######## Num of channels/features - evolution ########

        # How the features in Encoder evolve (number of features)
        self.encoder_features = [self.lift_dim]
        for i in range(self.N_layers):
            self.encoder_features.append(2 ** i * self.channel_multiplier)

        # How the features in Decoder evolve (number of features)
        self.decoder_features_in = self.encoder_features[1:]
        self.decoder_features_in.reverse()
        self.decoder_features_out = self.encoder_features[:-1]
        self.decoder_features_out.reverse()

        for i in range(1, self.N_layers):
            # Pad the outputs of the resnets (we must multiply by 2 then)
            self.decoder_features_in[i] = 2*self.decoder_features_in[i]

        ######## Spatial sizes of channels - evolution ########

        self.encoder_sizes = []
        self.decoder_sizes = []
        for i in range(self.N_layers + 1):
            self.encoder_sizes.append(size // 2 ** i)
            self.decoder_sizes.append(size // 2 ** (self.N_layers - i))

        ######## Define Lift and Project blocks ########

        self.lift = LiftProjectBlock(in_channels=in_dim,
                                     out_channels=self.encoder_features[0],
                                     size=size)

        self.project = LiftProjectBlock(in_channels=self.encoder_features[0] + self.decoder_features_out[-1],
                                        out_channels=out_dim,
                                        size=size)

        ######## Define Encoder, ED Linker and Decoder networks ########

        self.encoder = nn.ModuleList([(CNOBlock(in_channels=self.encoder_features[i],
                                                out_channels=self.encoder_features[i+1],
                                                in_size=self.encoder_sizes[i],
                                                out_size=self.encoder_sizes[i+1],
                                                use_bn=use_bn))
                                      for i in range(self.N_layers)])

        # After the ResNets are executed, the sizes of encoder and decoder might
        #  not match (if out_size>1)
        # We must ensure that the sizes are the same, by aplying CNO Blocks
        self.ED_expansion = nn.ModuleList([(CNOBlock(in_channels=self.encoder_features[i],
                                                     out_channels=self.encoder_features[i],
                                                     in_size=self.encoder_sizes[i],
                                                     out_size=self.decoder_sizes[self.N_layers - i],
                                                     use_bn=use_bn))
                                           for i in range(self.N_layers + 1)])

        self.decoder = nn.ModuleList([(CNOBlock(in_channels=self.decoder_features_in[i],
                                                out_channels=self.decoder_features_out[i],
                                                in_size=self.decoder_sizes[i],
                                                out_size=self.decoder_sizes[i+1],
                                                use_bn=use_bn))
                                      for i in range(self.N_layers)])

        ####################### Define ResNets Blocks ###########################

        # Here, we define ResNet Blocks.

        # Operator UNet:
        # Outputs of the middle networks are patched (or padded) to corresponding
        # sets of feature maps in the decoder

        self.res_nets = []
        self.N_res = int(N_res)
        self.N_res_neck = int(N_res_neck)

        # Define the ResNet networks (before the neck)
        for l in range(self.N_layers):
            self.res_nets.append(ResNet(channels=self.encoder_features[l],
                                        size=self.encoder_sizes[l],
                                        num_blocks=self.N_res,
                                        use_bn=use_bn))

        self.res_net_neck = ResNet(channels=self.encoder_features[self.N_layers],
                                   size=self.encoder_sizes[self.N_layers],
                                   num_blocks=self.N_res_neck,
                                   use_bn=use_bn)

        self.res_nets = torch.nn.Sequential(*self.res_nets)

    def forward(self, x):

        x = self.lift(x)  # Execute Lift
        skip = []

        # Execute Encoder
        for i in range(self.N_layers):

            # Apply ResNet & save the result
            y = self.res_nets[i](x)
            skip.append(y)

            # Apply (D) block
            x = self.encoder[i](x)

        # Apply the deepest ResNet (bottle neck)
        x = self.res_net_neck(x)

        # Execute Decode
        for i in range(self.N_layers):

            # Apply (I) block (ED_expansion) & cat if needed
            if i == 0:
                # BottleNeck : no cat
                x = self.ED_expansion[self.N_layers - i](x)
            else:
                x = torch.cat(
                    (x, self.ED_expansion[self.N_layers - i](skip[-i])), 1)

            # Apply (U) block
            x = self.decoder[i](x)

        # Cat & Execute Projetion
        x = torch.cat((x, self.ED_expansion[0](skip[0])), 1)
        x = self.project(x)

        return x
