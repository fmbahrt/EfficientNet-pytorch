import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_output_image_size(input_image_size, stride):
    """Calculates the output image size when using Conv2dSamePadding with a stride.
    Args:
        input_image_size (int, tuple or list): Size of input image.
        stride (int, tuple or list): Conv2d operation's stride.
    Returns:
        output_image_size: A list [H,W].
    """
    def get_width_and_height_from_size(x):
        """Obtain height and width from x.
        Args:
            x (int, tuple or list): Data size.
        Returns:
            size: A tuple or list (H,W).
        """
        if isinstance(x, int):
            return x, x
        if isinstance(x, list) or isinstance(x, tuple):
            return x
        else:
            raise TypeError()

    if input_image_size is None:
        return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]

class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    # Kudos : github.com/lukemelas/EfficientNet-PyTorch
    # With the same calculation as Conv2dDynamicSamePadding

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w - pad_w // 2, pad_w - pad_w // 2,
                                                pad_h - pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

Conv2d = Conv2dStaticSamePadding

class SEBlock(nn.Module):

    """Sweeet squeeze and excitation module!"""

    def __init__(self, in_channel, ratio=0.25):
        super(SEBlock, self).__init__()

        self.hid_dim = max(1, int(in_channel * ratio))

        self._pool = nn.AdaptiveAvgPool2d(1)
        self._ln1  = nn.Linear(in_channel, self.hid_dim)
        self._ln2  = nn.Linear(self.hid_dim, in_channel)

        self._act1 = nn.ReLU()
        self._act2 = nn.Hardsigmoid() # we dont not like sigmoid hihi :^)

    def forward(self, x):
        b, c, _, _ = x.size() # Squeeze doesnt work if batch dim = 1 :---(

        # FC part
        z = self._pool(x).view(b, c)
        z = self._ln1(z)
        z = self._act1(z)
        z = self._ln2(z)
        z = self._act2(z).view(b, c, 1, 1)

        # Now weigh each channel
        return x * z

class DropConnect(nn.Module):

    def __init__(self, drop_rate):
        assert 0 <= drop_rate <= 1, "drop_rate must be in the range [0, 1]"

        super(DropConnect, self).__init__()

        self.drop_rate = drop_rate

    def forward(self, x):
        if self.training:
            batch_size = x.shape[0]
            keep_prob = 1 - self.drop_rate

            # This is perhaps a weird way to produce the mask and can perhaps be
            # done better...
            # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
            random_tensor = keep_prob
            random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype,
                                        device=x.device)
            binary_tensor = torch.floor(random_tensor)

            # From a model ensemble perspective we recalibrate the input tensor
            # during training such that we do not have to do it during testing.
            x = x / keep_prob * binary_tensor
        return x

class InvertedResidualBlock(nn.Module):

    def __init__(self,
                 inp,
                 oup,
                 expand_ratio,
                 kernel_size,
                 stride,
                 se_ratio=0.25,
                 image_size=None,
                 skip_connect=True,
                 drop_rate=0.0,
                 bn_mom=0.99,
                 bn_eps=1e-3,
                 swish=False):
        super(InvertedResidualBlock, self).__init__()

        # BatchNorm parameters
        bn_mom = 1 - bn_mom

        hiddendim = inp * expand_ratio

        self.inp = inp
        self.oup = oup
        self.hip = hiddendim
        self.stride = stride
        self.skip_connect = skip_connect
        self.expand_ratio = expand_ratio

        se = ((se_ratio is not None) and (0 < se_ratio <=1))

        # Drop connect
        self._drop_connect = DropConnect(drop_rate) if drop_rate>0 else nn.Identity()

        # Channel Expansion
        self._conv_exp = Conv2d(inp, hiddendim,
                                kernel_size=1,
                                stride=1,
                                bias=False,
                                image_size=image_size)
        self._bn_exp = nn.BatchNorm2d(hiddendim, momentum=bn_mom, eps=bn_eps)
        self._act_exp = nn.Hardswish() if swish else nn.ReLU()

        self._expand_block = nn.Sequential(
            self._conv_exp,
            self._bn_exp,
            self._act_exp
        )

        # no reason to expand if expand ratio is 1
        self._expand = self._expand_block if expand_ratio != 1 else nn.Identity()

        # Depthwise Convolution
        self._depth = Conv2d(in_channels=hiddendim, out_channels=hiddendim,
                                groups=hiddendim, kernel_size=kernel_size,
                                stride=stride, image_size=image_size, bias=False)
        self._bn_depth = nn.BatchNorm2d(hiddendim, momentum=bn_mom, eps=bn_eps)
        self._act_depth = nn.Hardswish() if swish else nn.ReLU()

        # Calculate new image size since stride is handled by depthwise
        image_size = calculate_output_image_size(image_size, self.stride)

        # Squeeze and excitation
        self._squeeze = SEBlock(hiddendim, ratio=se_ratio) if se else nn.Identity()

        # Projection
        self._proj = Conv2d(in_channels=hiddendim, out_channels=oup,
                            kernel_size=1, stride=1, bias=False,
                            image_size=image_size)
        self._bn_proj = nn.BatchNorm2d(oup, momentum=bn_mom, eps=bn_eps)

    def forward(self, x):
        ins = x

        x = self._expand(x)

        # Depthwise Conv
        x = self._depth(x)
        x = self._bn_depth(x)
        x = self._act_depth(x)

        # SE
        x = self._squeeze(x)

        # Project
        x = self._proj(x)
        x = self._bn_proj(x)

        # If input dim equal output dim then we use a residual connection
        #  as stated in mobilenetv2 paper
        if self.skip_connect and self.inp == self.oup and self.stride == 1:
            # We uses drop connect if a skip connection exists
            x = ins + self._drop_connect(x)

        return x

if __name__ == '__main__':
    inz = torch.randn(4, 3, 7, 7)

    m = DropConnect(drop_rate=0.0)
    x = m(inz)
