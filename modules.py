import numpy
import torch
import torch.nn as nn
from ml_modules import FCBlock, CNNEncoder, CNNEncoderVGG16, EqualConv2d, EqualLinear
from pytorch3d.transforms import  Rotate, rotation_6d_to_matrix, matrix_to_rotation_6d, matrix_to_quaternion
import numpy as np
from src.filter_utils import GaussianPyramid
from ml_modules_progresive import EncoderProgressive



def weights_init(m, args):
    if isinstance(m, nn.Conv2d):

        if m.weight is not None:
            torch.nn.init.kaiming_normal_(m.weight, a=args.leak_value)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)

    if isinstance(m, nn.Linear):
        if m.weight is not None:
            torch.nn.init.kaiming_normal_(m.weight, a=args.leak_value)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)


class Discriminator(torch.nn.Module):
    def __init__(self, args):
        ''' 6: simple conv network with max pooling'''
        super(Discriminator, self).__init__()
   
        K = args.num_channel_Discriminator  # num channels
        N = args.num_N_Discriminator  # penultimate features
        numConvs = args.num_layer_Discriminator

        # first one halves the number of numbers, then multiplies by K
        # interval convolutions, each halves the number of values (because channels double)
        if args.gaussian_pyramid:
            self.gaussian_filters = GaussianPyramid(kernel_size=11, kernel_variance=1., num_octaves=args.num_octaves,
                                                    octave_scaling=1.414)
            num_additional_channels = args.num_octaves
        else:
            num_additional_channels = 0
            
            
        self.convs = nn.ModuleList(
            [torch.nn.Sequential(
                torch.nn.Conv2d(2 ** (i) * K ** (i > 0) +num_additional_channels*(i==0), 2 ** (i + 1) * K, kernel_size=3,
                                stride=1, padding=1),
                torch.nn.LeakyReLU(args.leak_value),
            torch.nn.MaxPool2d(kernel_size=2))
                for i in range(numConvs)]
        )

        # todo: have to think about how to handle this
        size = 2*2**(numConvs)

        # flatten down to N numbers, then 1 number
        # size=K * size**2 * 2**numConvs / 4**numConvs

        input = torch.zeros(1, 1 +num_additional_channels, int(size), int(size))
        with torch.no_grad():
            for conv in self.convs:
                input = conv(input)

        self.fully = torch.nn.Sequential(
            torch.nn.Linear(np.prod(input.size()), N),
            torch.nn.LeakyReLU(args.leak_value),
            torch.nn.Linear(N, N),
            torch.nn.LeakyReLU(args.leak_value)
            # torch.nn.ReLU()
        )

        self.linear = torch.nn.Linear(N, 1)
        self.args = args
        self.normalizer=torch.nn.InstanceNorm2d(num_features=1, momentum=0.0)



    def forward(self, input_image):

            output = input_image

            if self.args.normalize_dis_input:
                output=self.normalizer(output)

            if self.args.gaussian_pyramid:
                output=self.gaussian_filters(output)



            for conv in self.convs:
                output = conv(output)
            self.cnn_output=output

            output_linear= self.fully(self.cnn_output.flatten(1))

            self.output = self.linear(output_linear)

            return output


class ExplicitVolume(torch.nn.Module):
    def __init__(self, config):
        super(ExplicitVolume, self).__init__()

        self.sidelen = config.sidelen
        sidelen=self.sidelen
        self.vol_shape = [self.sidelen] * 3
        self.vol = nn.Parameter(0.01 * torch.rand(self.vol_shape, dtype=torch.float32),
                                requires_grad=True)

        # lincoords = torch.linspace(-1. / np.sqrt(3.), 1. / np.sqrt(3.), sidelen)  # assume square volume
        lincoords = torch.linspace(-1., 1., sidelen)  # assume square volume
        # TODO: Figure why we need to flip X and Y in meshgrid
        [X, Y, Z] = torch.meshgrid([lincoords, lincoords, lincoords])
        # Y = torch.flip(Y, dims=[1])
        coords = torch.stack([Y, X, Z], dim=-1)
        self.register_buffer('vol_coords', coords.reshape(-1, 3))
        
        self.downsample= config.ProjectionSize < config.sidelen
        
        self.config=config
        
    # project and rotate
    def forward(self, rotmat, scalar, proj_axis=-1):
        batch_sz = rotmat.shape[0]

        t = Rotate(rotmat, device=self.vol_coords.device)
        rot_vol_coords = t.transform_points(self.vol_coords.repeat(batch_sz, 1, 1))
        # one could use torch.relu for positivity constraint

        rot_vol = torch.nn.functional.grid_sample(self.vol.repeat(batch_sz, 1, 1, 1, 1),  # = (Batch, C,D,H,W)
                                                  rot_vol_coords[:, None, None, :, :],
                                                  align_corners=True)
        projection = torch.sum(rot_vol.reshape(batch_sz, self.sidelen,
                                               self.sidelen,
                                               self.sidelen),
                               dim=proj_axis)[:, None, :, :]
        projection_clean = projection  # add a dummy channel (for consistency w/ img fmt)
        # --> B,C,H,W
        if self.config.noise:
            projection = projection_clean + torch.exp(scalar[0]) * torch.randn_like(projection_clean)
            if self.downsample:
                projection_clean=downsample_fourier_crop(projection_clean, size=self.config.ProjectionSize)
                projection=downsample_fourier_crop(projection, size=self.config.ProjectionSize)
   
        projection=torch.exp(scalar[1])*projection
        projection_clean=torch.exp(scalar[1])*projection_clean
        return projection, projection_clean

    def make_volume(self):
        return self.vol.detach()



class EncoderCNNFactory:
        """Class to instantiate projector from a factory of choices."""

        def get_cnn_encoder(args, num_additional_channels):
            if args.multires:
                cnn_encoder = EncoderProgressive(args, in_channels=1 + num_additional_channels, batch_norm=args.encoder_batch_norm)
            elif args.encoder_type == "cryoposenet":

                    cnn_encoder = CNNEncoder(in_channels=1 + num_additional_channels,
                                                  feature_channels=args.encoder_conv_layers,
                                                  padding=True,
                                                  batch_norm=args.encoder_batch_norm,
                                                  max_pool=args.encoder_max_pool,
                                                  lr_equalization=args.encoder_equalized_lr,
                                                  global_avg_pool=True)

            else:

                    cnn_encoder = CNNEncoderVGG16(in_channels=1 + num_additional_channels,
                                                       batch_norm=args.encoder_batch_norm,
                                                       pretrained=False,
                                                       flip_images=args.flip_images)

            return cnn_encoder

class EncoderMLPFactor:
    def get_mlp_encoder( args, latent_code_size):
        orientation_mlp_1 = FCBlock(in_features=latent_code_size,
                                         out_features=args.regressor_orientation_layers[-1],
                                         features=args.regressor_orientation_layers[:-1],
                                         nonlinearity='relu', last_nonlinearity='relu',
                                         batch_norm=args.encoder_batch_norm,
                                         equalized=args.encoder_equalized_lr,
                                         dropout=False)
        orientation_mlp_2 = FCBlock(in_features=args.regressor_orientation_layers[-1],
                                         out_features=args.orientation_dims,
                                         features=[],
                                         nonlinearity='relu',
                                         last_nonlinearity='tanh',
                                         batch_norm=args.encoder_batch_norm,
                                         equalized=args.encoder_equalized_lr,
                                         dropout=False)

        return orientation_mlp_1, orientation_mlp_2


class SO3paramterizationFactory:
    def get_so3_parametrization(args):
        if args.so3_parameterization=="s2s2":
               latent_to_rot3d_fn = rotation_6d_to_matrix
               return latent_to_rot3d_fn
        elif args.so3_parameterization=="positive_s2s2":
                latent_to_rot3d_fn = rotation_6d_to_matrix

                pos_latent=lambda x: latent_to_rot3d_fn(torch.nn.functional.relu(x) )
                return pos_latent

        else:
            raise NotImplementedError("so3_parameterization in encoder not implemented!!")

class Encoder(torch.nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.unet=None
        num_additional_channels = 0
        if args.gaussian_pyramid:
            self.gaussian_filters = GaussianPyramid(kernel_size=11, kernel_variance=1., num_octaves=args.num_octaves,
                                                    octave_scaling=1.414)
            num_additional_channels = args.num_octaves
            
        if args.use_3d_volume_encoder:
            self.projection_images=torch.zeros(1,args.ProjectionSize,args.ProjectionSize, args.ProjectionSize)
            num_additional_channels=num_additional_channels+args.ProjectionSize


        self.cnn_encoder=EncoderCNNFactory.get_cnn_encoder(args, num_additional_channels)
        cnn_encoder_out_shape = self.cnn_encoder.get_out_shape(args.map_shape, args.map_shape)
        latent_code_size = torch.prod(torch.tensor(cnn_encoder_out_shape))
        print(self.cnn_encoder)
        self.orientation_mlp_1, self.orientation_mlp_2 = EncoderMLPFactor.get_mlp_encoder(args, latent_code_size)

        self.latent_to_rot3d_fn=SO3paramterizationFactory.get_so3_parametrization(args)

        self.normalizer=torch.nn.InstanceNorm2d(num_features=1, momentum=0.0)


    def forward(self, input_image):

        input_image=self.normalizer(input_image)/self.args.factor_normalization

        if self.args.gaussian_pyramid:
            input_image = self.gaussian_filters(input_image)

        if self.args.use_3d_volume_encoder:
            input_image=torch.cat([input_image,self.projection_images.repeat(input_image.shape[0],1,1,1)],1)

        self.output_cnn=self.cnn_encoder(input_image)
        
        output = torch.flatten(self.output_cnn, start_dim=1)
        self.output=self.orientation_mlp_2(self.orientation_mlp_1(output))
 
        return self.latent_to_rot3d_fn(self.output)



from ml_modules import SIREN
 
class ImplicitVolume(torch.nn.Module):
    def __init__(self, sidelen):
        super(ImplicitVolume, self).__init__()

        self.sidelen = sidelen  # only used for sampling

        lincoords = torch.linspace(-1., 1., sidelen)  # assume square volume
        [X, Y, Z] = torch.meshgrid([lincoords, lincoords, lincoords])
        coords = torch.stack([Y, X, Z], dim=-1)
        self.register_buffer('vol_coords', coords.reshape(-1, 3))


            # The dictionary params_implicit could be used to easily change the parameters of the SIREN
        self.vol = SIREN(in_features=3, out_features=1,
                         num_hidden_layers=3, hidden_features=256,
                         outermost_linear=True, w0=30)
        self.pe = None
  
    def forward(self, rotmat, local_def={}):
        batch_sz = rotmat.shape[0]
        rot_vol_coords = torch.bmm(self.vol_coords.repeat(batch_sz, 1, 1), rotmat)  # --> Batch, sidelen^3, 3

        rot_vol = self.vol(rot_vol_coords).reshape(batch_sz, self.sidelen,
                                                   self.sidelen,
                                                   self.sidelen)
        projection = torch.sum(rot_vol, axis=-1)
        projection = projection[:, None, :, :]

        return projection

    def make_volume(self):
   
            coords = self.vol_coords
            if self.pe is not None:
                coords = self.pe(coords)
            exp_vol = self.vol(coords).reshape(self.sidelen, self.sidelen, self.sidelen, 1)

            return exp_vol


class BiDiscriminator(torch.nn.Module):
    def __init__(self, args):
        super(BiDiscriminator, self).__init__()
        self.Fourier = args.FourierDiscriminator
        K = args.num_channel_Discriminator  # num channels
        N = args.num_N_Discriminator  # penultimate features
        numConvs = args.num_layer_Discriminator

        # first one halves the number of numbers, then multiplies by K
        # interval convolutions, each halves the number of values (because channels double)

        self.convs = nn.ModuleList(
            [torch.nn.Sequential(
                torch.nn.Conv2d(2 ** (i) * K ** (i > 0) + 2 * self.Fourier * (i == 0), 2 ** (i + 1) * K, kernel_size=3,
                                stride=1, padding=1),
                torch.nn.MaxPool2d(kernel_size=2),
                torch.nn.LeakyReLU(args.leak_value))
                for i in range(numConvs)]
        )

        # todo: have to think about how to handle this
        size = args.ProjectionSize

        # flatten down to N numbers, then 1 number
        # size=K * size**2 * 2**numConvs / 4**numConvs

        input = torch.zeros(1, 1 + self.Fourier * 2, int(size), int(size))
        with torch.no_grad():
            for conv in self.convs:
                input = conv(input)

        self.fully_linear_image = torch.nn.Sequential(
            torch.nn.Linear(np.prod(input.size()), N),
            torch.nn.LeakyReLU(args.leak_value),

            # torch.nn.ReLU()
        )

        self.fully_linear_rotmat = torch.nn.Sequential(
            torch.nn.Linear(9, 4 * N),
            torch.nn.LeakyReLU(args.leak_value),
            torch.nn.Linear(4 * N, 4 * N),
            torch.nn.LeakyReLU(args.leak_value),
            torch.nn.Linear(4 * N, N),
            torch.nn.LeakyReLU(args.leak_value),
            torch.nn.Linear(N, N),
            torch.nn.LeakyReLU(args.leak_value)

        )
        self.LayerNormImage = torch.nn.LayerNorm(N)
        self.LayerNormRotmat = torch.nn.LayerNorm(N)

        self.coupling = torch.nn.Sequential(
            torch.nn.Linear(2 * N, 4 * N),
            torch.nn.LeakyReLU(args.leak_value),
            torch.nn.Linear(4 * N, 4 * N),
            torch.nn.LeakyReLU(args.leak_value),
            torch.nn.Linear(4 * N, N),
            torch.nn.LeakyReLU(args.leak_value),
            torch.nn.Linear(N, N),
            torch.nn.LeakyReLU(args.leak_value),

        )

        self.linear = torch.nn.Linear(N, 1)
        self.linear1 = torch.nn.Linear(N, 1)
        self.args = args

        print("Decoupled discriminator for SO3 rotmat and image parts")

    def split(self, data):

        image = data["samps"]
        rotmat = data["rotmat"]

        return image, rotmat

    def forward(self, data):

        image, rotmat = self.split(data)
        output = image
        if self.Fourier:
            output = torch.cat([output, SpaceToFourier(output, signal_dim=2).permute(0, 4, 2, 3, 1).squeeze(-1)], 1)

        for conv in self.convs:
            output = conv(output)

        output_image = self.fully_linear_image(output.reshape(output.shape[0], -1))
        output_rotmat = self.fully_linear_rotmat(rotmat.reshape(rotmat.shape[0], -1))

        self.output_image = self.LayerNormImage(output_image)
        self.output_rotmat = self.LayerNormRotmat(output_rotmat)

        self.output_image.retain_grad()
        self.output_rotmat.retain_grad()

        output = torch.cat([self.output_image, self.output_rotmat], -1)

        # output=self.linear(self.output_image)-self.linear(self.output_rotmat)
        return output


# class Encoder

class RotDiscriminator(torch.nn.Module):
    def __init__(self, args):
        super(RotDiscriminator, self).__init__()
        N = args.num_N_Discriminator
        self.fully_linear_rotmat = torch.nn.Sequential(
            torch.nn.Linear(9, N),
            torch.nn.LeakyReLU(args.leak_value),
            torch.nn.Linear(N, N),
            torch.nn.LeakyReLU(args.leak_value),
            torch.nn.Linear(N, N),
            torch.nn.LeakyReLU(args.leak_value),
            torch.nn.Linear(N, 1),

        )

        self.args = args

        print("Discriminator for imposing SO3 on rotmat")

    def split(self, data):

        image = data["samps"]
        rotmat = data["rotmat"]
        return image, rotmat

    def forward(self, data):
        if isinstance(data, dict):
            image, rotmat = self.split(data)
        else:
            rotmat = data

        # rotmat6d=matrix_to_rotation_6d(rotmat)
        output_rotmat = self.fully_linear_rotmat(rotmat.reshape(rotmat.shape[0], -1))
        self.output_rotmat = output_rotmat

        return output_rotmat


class FCEncoder(torch.nn.Module):
    def __init__(self, args):
        super(FCEncoder, self).__init__()
        self.args = args

        self.orientation_mlp_1 = FCBlock(in_features=128 * 16,
                                         out_features=args.regressor_orientation_layers[-1],
                                         features=args.regressor_orientation_layers[:-1],
                                         nonlinearity='relu', last_nonlinearity='relu')
        self.orientation_mlp_2 = FCBlock(in_features=args.regressor_orientation_layers[-1],
                                         out_features=args.orientation_dims,
                                         features=[],
                                         nonlinearity=[],
                                         last_nonlinearity='tanh')

        if self.args.so3_parameterization == "s2s2":
            self.latent_to_rot3d_fn = rotation_6d_to_matrix

    def forward(self, input):
        output = torch.flatten(input, start_dim=1)
        self.output = self.orientation_mlp_2(self.orientation_mlp_1(output))

        return self.latent_to_rot3d_fn(self.output)