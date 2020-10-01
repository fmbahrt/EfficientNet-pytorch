import torch
import torch.nn as nn

import math
import collections

from .layers import InvertedResidualBlock, Conv2d, calculate_output_image_size

# Kindly stolen from : https://github.com/qubvel/efficientnet
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])

BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

DEFAULT_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
]

MODEL_PARAMS = {
    # phi:   width,depth,res,dropout
    0: (1.0, 1.0, 224, 0.2),
    1: (1.0, 1.1, 240, 0.2),
    2: (1.1, 1.2, 260, 0.3),
    3: (1.2, 1.4, 300, 0.3),
    4: (1.4, 1.8, 380, 0.4),
    5: (1.6, 2.2, 456, 0.4),
    6: (1.8, 2.6, 528, 0.5),
    7: (2.0, 3.1, 600, 0.5),
}

def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on width multiplier."""

    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""

    return int(math.ceil(depth_coefficient * repeats))



class EfficientNet(nn.Module):

    """ TODO
    arguments:
        drop_connect_rate: float, survival probability for the last conv block
    """

    def __init__(self,
                 phi=0,
                 num_classes=1000,
                 drop_connect_rate=0.2,
                 bn_mom=0.99,
                 bn_eps=1e-3,
                 with_relu=True):

        super(EfficientNet, self).__init__()

        # BatchNorm parameters
        bn_mom = 1 - bn_mom

        width_coefficient, depth_coefficient, image_size, droprate = MODEL_PARAMS[phi]

        depth_divisor=8

        block_args = DEFAULT_BLOCKS_ARGS

        # Stem
        stem_filters = round_filters(32, width_coefficient, depth_divisor)
        self._stem_conv = Conv2d(3, stem_filters,
                                 kernel_size=3,
                                 stride=2,
                                 bias=False,
                                 image_size=image_size)
        self._stem_bn = nn.BatchNorm2d(stem_filters, momentum=bn_mom,
                                       eps=bn_eps)
        self._stem_act = nn.Hardswish()

        self._stem = nn.Sequential(
            self._stem_conv,
            self._stem_bn,
            self._stem_act
        )

        # Update image_size
        image_size = calculate_output_image_size(image_size, 2)

        num_blocks_total = sum(round_repeats(block_arg.num_repeat,
                                             depth_coefficient) for block_arg in block_args)
        # Use relu for the first half of the network
        #  it is efficientnet after all
        relu_half = num_blocks_total // 2

        blocks = []
        last_block = None
        # Block num is used to calculate drop_connect_rate. We start at 0 such
        # that the first block has a survival probability of 1
        block_num = 0
        for idx, block_arg in enumerate(block_args):

            # negate bool because I am weird
            use_swish = not(((block_num + 1) >= relu_half) and with_relu)

            # Update block input and output filters based on depth multiplier.
            block_arg = block_arg._replace(
                input_filters=round_filters(block_arg.input_filters,
                                            width_coefficient, depth_divisor),
                output_filters=round_filters(block_arg.output_filters,
                                             width_coefficient, depth_divisor),
                num_repeat=round_repeats(block_arg.num_repeat, depth_coefficient)
            )

            last_block = block_arg

            # Calculate drop_connect_rate linearly
            drop_rate = drop_connect_rate * block_num / num_blocks_total

            # Construct first block
            block = InvertedResidualBlock(
                block_arg.input_filters,
                block_arg.output_filters,
                block_arg.expand_ratio,
                block_arg.kernel_size,
                block_arg.strides,
                se_ratio=block_arg.se_ratio,
                skip_connect=block_arg.id_skip,
                image_size=image_size,
                drop_rate=drop_rate,
                swish=use_swish
            )
            blocks.append(block)
            image_size = calculate_output_image_size(image_size,
                                                     block_arg.strides)
            block_num += 1
            if block_arg.num_repeat > 1:
                # only the first block takes care of strid and filter size increase
                block_arg = block_arg._replace(
                    input_filters=block_arg.output_filters,
                    strides=[1,1]
                )


                for bidx in range(block_arg.num_repeat - 1):

                    # Calculate drop_connect_rate linearly
                    drop_rate = drop_connect_rate * block_num / num_blocks_total

                    block = InvertedResidualBlock(
                        block_arg.input_filters,
                        block_arg.output_filters,
                        block_arg.expand_ratio,
                        block_arg.kernel_size,
                        block_arg.strides,
                        se_ratio=block_arg.se_ratio,
                        skip_connect=block_arg.id_skip,
                        image_size=image_size,
                        drop_rate=drop_rate,
                        swish=use_swish
                    )
                    blocks.append(block)
                    block_num += 1
        self._blocks = nn.Sequential(*blocks)

        # Classification
        in_filters  = last_block.output_filters
        out_filters = round_filters(1280, width_coefficient, depth_divisor)

        self._class_conv = Conv2d(in_filters, out_filters, kernel_size=1,
                                 bias=False, stride=1, image_size=image_size)
        self._class_bn = nn.BatchNorm2d(out_filters, momentum=bn_mom,
                                        eps=bn_eps)

        self._head = nn.Sequential(
            self._class_conv,
            self._class_bn
        )

        # Final layer
        self._pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(drop_connect_rate)
        self._logits  = nn.Linear(out_filters, num_classes)

    def forward(self, x):
        x = self._stem(x)
        x = self._blocks(x)
        x = self._head(x)

        # Final Layer
        x = self._pooling(x)
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self._logits(x)

        return x

if __name__ == '__main__':
    import time
    inz = torch.randn(32, 3, 260, 260)

    model = EfficientNet(phi=0, with_relu=False)

    start = time.time()
    for i in range(5):
        x = model(inz)
        print(x.shape)
    end = time.time() - start

    print(end)

