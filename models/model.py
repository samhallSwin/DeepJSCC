import tensorflow as tf
import tensorflow_compression as tfc
from models.channellayer import RayleighChannel, AWGNChannel, RicianChannel
#from models.vitblock import VitBlock

class deepJSCC(tf.keras.Model):
    def __init__(self, has_gdn=True,
                 num_symbols=512, snrdB=25, channel='AWGN', input_size=32):
        
        super().__init__()
        if has_gdn:
            gdn_func=tfc.layers.GDN()
            igdn_func=tfc.layers.GDN(inverse=True)
        else:
            gdn_func=tf.keras.layers.Lambda(lambda x: x)
            igdn_func=tf.keras.layers.Lambda(lambda x: x)

        block_types = 'CCCCCC'
        filters = [256, 256, 256, 256, 256, 256]
        num_blocks = [1, 1, 3, 3, 1, 1]

        assert len(block_types) == len(filters) == len(num_blocks) == 6, \
               "length of block_types, filters, num_blocks should be 6"
        self.encoder = Encoder(
            block_types[:3],
            filters[:3],
            num_blocks[:3],
            num_symbols,
            input_size=input_size,
            gdn_func=gdn_func
        )

        self.channel = AWGNChannel(snrdB)

        '''if channel == 'Rayleigh':
            self.channel = RayleighChannel(snrdB)
        elif channel == 'AWGN':
            self.channel = AWGNChannel(snrdB)
        elif channel == 'Rician':
            self.channel = RicianChannel(snrdB, k=2)
        else:
            self.channel = tf.identity'''

        self.decoder = Decoder(
            block_types[3:],
            filters[3:],
            num_blocks[3:],
            input_size=input_size,
            gdn_func=igdn_func
        )
    
    def call(self, x):
        x = self.encoder(x)
        x = self.channel(x)
        x = self.decoder(x)

        return x
    
class Encoder(tf.keras.layers.Layer):
    def __init__(self, block_types, filters, num_blocks,
                 num_symbols, gdn_func=None, input_size = 32, **kwargs):
        super().__init__()

        if input_size==64:
            self.layers = [
                # 64 x 64 input
                build_blocks(0, block_types, num_blocks, filters, 64, kernel_size=9, stride=2, gdn_func=gdn_func),
                # 32 x 32 input
                build_blocks(0, block_types, num_blocks, filters, 32, kernel_size=9, stride=2, gdn_func=gdn_func),
                # downsampled to 16 x 16
                build_blocks(1, block_types, num_blocks, filters, 16, kernel_size=5, stride=2, gdn_func=gdn_func),
                # downsampled to 8 x 8
                build_blocks(2, block_types, num_blocks, filters, 8, kernel_size=5, gdn_func=gdn_func),
                # to constellation
                tf.keras.layers.Conv2D(
                    filters=num_symbols // 8 // 8 * 2,
                    # current spatial dimension is 8 x 8
                    # and 2 for iq dimension
                    kernel_size=1
                )
            ]
        elif input_size==32:
            self.layers = [
                # 32 x 32 input
                build_blocks(0, block_types, num_blocks, filters, 32, kernel_size=9, stride=2, gdn_func=gdn_func),
                # downsampled to 16 x 16
                build_blocks(1, block_types, num_blocks, filters, 16, kernel_size=5, stride=2, gdn_func=gdn_func),
                # downsampled to 8 x 8
                build_blocks(2, block_types, num_blocks, filters, 8, kernel_size=5, gdn_func=gdn_func),
                # to constellation
                tf.keras.layers.Conv2D(
                    filters=num_symbols // 8 // 8 * 2,
                    # current spatial dimension is 8 x 8
                    # and 2 for iq dimension
                    kernel_size=1
                )
            ]
        else:
            print('Incorrect input size!')

    def call(self, x):
        for sublayer in self.layers:
            x = sublayer(x)
        
        b, h, w, c = x.shape
        x = tf.reshape(x, (-1, h*w*c//2, 2))
        return x
    

    def get_config(self):
        config = super().get_config()
        return config
    

class Decoder(tf.keras.layers.Layer):
    def __init__(self, block_types, filters, num_blocks, gdn_func=None, input_size=32, **kwargs):
        super().__init__()

        if input_size==64:
            self.layers = [
                # 8 x 8 input
                build_blocks(0, block_types, num_blocks, filters, 8, kernel_size=5, gdn_func=gdn_func),
                # upsampled to 16 x 16
                tf.keras.layers.Resizing(16, 16),
                build_blocks(1, block_types, num_blocks, filters, 16, kernel_size=5, gdn_func=gdn_func),
                # upsampled to 32 x 32
                tf.keras.layers.Resizing(32, 32),
                build_blocks(2, block_types, num_blocks, filters, 32, kernel_size=9, gdn_func=gdn_func),
                # upsampled to 64 x 64
                tf.keras.layers.Resizing(64, 64),
                build_blocks(2, block_types, num_blocks, filters, 64, kernel_size=9, gdn_func=gdn_func),
                # to image
                tf.keras.layers.Conv2D(
                    filters=3,
                    kernel_size=1,
                    activation='sigmoid'
                )
            ] 
        elif input_size==32:  
            self.layers = [
                # 8 x 8 input
                build_blocks(0, block_types, num_blocks, filters, 8, kernel_size=5, gdn_func=gdn_func),
                # upsampled to 16 x 16
                tf.keras.layers.Resizing(16, 16),
                build_blocks(1, block_types, num_blocks, filters, 16, kernel_size=5, gdn_func=gdn_func),
                # upsampled to 32 x 32
                tf.keras.layers.Resizing(32, 32),
                build_blocks(2, block_types, num_blocks, filters, 32, kernel_size=9, gdn_func=gdn_func),
                # to image
                tf.keras.layers.Conv2D(
                    filters=3,
                    kernel_size=1,
                    activation='sigmoid'
                )
            ]
        else:
            print('Incorrect input size!')


    def call(self, x):
        b, c, _ = x.shape
        x = tf.reshape(x, (-1, 8, 8, c*2//64))

        for sublayer in self.layers:
            x = sublayer(x)
        return x
    

    def get_config(self):
        config = super().get_config()
        return config

def build_blocks(layer_idx, block_types, num_blocks, filters, spatial_size, kernel_size=5, stride=1, gdn_func=None):
    assert block_types[layer_idx] in ('C', 'V'), "layer type should be either C or V"

    if block_types[layer_idx] == 'C':
        return build_conv(
            repetition=num_blocks[layer_idx],
            filter_size=filters[layer_idx],
            kernel_size=kernel_size,
            stride=stride,
            gdn_func=gdn_func)
    else:
        return build_conv(
            repetition=num_blocks[layer_idx],
            filter_size=filters[layer_idx],
            kernel_size=kernel_size,
            stride=stride,
            gdn_func=gdn_func)
    
def build_conv(repetition, filter_size, kernel_size=5, stride=1, gdn_func=None):
    x = tf.keras.Sequential()
    for i in range(repetition):
        s = stride if i == 0 else 1
        x.add(tfc.SignalConv2D(
                filter_size,
                kernel_size,
                corr=True,
                strides_down=s,
                padding="same_zeros",
                use_bias=True,
        ))
        if gdn_func:
            x.add(gdn_func)
        x.add(tf.keras.layers.PReLU(shared_axes=[1, 2]))
    return x


