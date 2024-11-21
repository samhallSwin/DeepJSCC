import tensorflow as tf
import tensorflow_compression as tfc
from models.channellayer import RayleighChannel, AWGNChannel, RicianChannel
from config import config

class deepJSCC(tf.keras.Model):
    def __init__(self):
        """
        Initialize the deepJSCC model using configuration from config.py.
        """
        super().__init__()
        gdn_func = tfc.layers.GDN() if config.architecture.get('has_gdn', True) else tf.keras.layers.Lambda(lambda x: x)
        igdn_func = tfc.layers.GDN(inverse=True) if config.architecture.get('has_gdn', True) else tf.keras.layers.Lambda(lambda x: x)

        # Encoder Initialization
        self.encoder = Encoder(config.architecture['encoder'], gdn_func)

        # Channel Model
        channel_type = config.architecture.get('channel', 'AWGN')
        snrdB = config.architecture.get('snrdB', 25)
        if channel_type == 'Rayleigh':
            self.channel = RayleighChannel(snrdB)
        elif channel_type == 'Rician':
            self.channel = RicianChannel(snrdB, k=2)
        else:
            self.channel = AWGNChannel(snrdB)

        # Decoder Initialization
        self.decoder = Decoder(config.architecture['decoder'], igdn_func)
    
    def call(self, x):
        x = self.encoder(x)
        x = self.channel(x)
        x = self.decoder(x)
        return x

class Encoder(tf.keras.layers.Layer):
    def __init__(self, params, gdn_func):
        super().__init__()
        layers = []
        for layer in params['layers']:
            layers.append(build_layer(layer, gdn_func))
        self.layers = tf.keras.Sequential(layers)

    def call(self, x):
        x = self.layers(x)
        b, h, w, c = x.shape  # Obtain spatial dimensions
        if h != 8 or w != 8:  # Ensure output matches channel layer's expectation
            raise ValueError(f"Expected encoder output spatial dimensions to be 8x8, got {h}x{w}. Check architecture config.")
        x = tf.reshape(x, (-1, h * w * c // 2, 2))  # Reshape to (batch_size, symbols, 2)
        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, params, gdn_func):
        super().__init__()
        layers = []
        for layer in params['layers']:
            layers.append(build_layer(layer, gdn_func))
        self.layers = tf.keras.Sequential(layers)

    def call(self, x):
        b, n, _ = x.shape  # Input shape from AWGNChannel
        spatial_dim = 8  # Expected spatial dimensions
        channels = n * 2 // (spatial_dim ** 2)
        if channels != 256:  # Ensure compatibility with architecture config
            raise ValueError(f"Expected decoder input channels to be 256, got {channels}. Check encoder output.")
        x = tf.reshape(x, (-1, spatial_dim, spatial_dim, channels))  # Reshape back to (batch_size, 8, 8, channels)
        x = self.layers(x)
        return x

def build_layer(layer_params, gdn_func):
    """
    Builds an individual layer dynamically based on its configuration.

    :param layer_params: Dictionary defining the layer type and its parameters.
    :param gdn_func: GDN function or identity layer.
    :return: A configured TensorFlow layer.
    """
    layer_type = layer_params['type']
    if layer_type == 'conv':
        return tf.keras.layers.Conv2D(
            filters=layer_params['filters'],
            kernel_size=layer_params['kernel_size'],
            strides=layer_params.get('strides', 1),
            padding='same',
            activation=layer_params.get('activation', None)
        )
    elif layer_type == 'resize':
        return tf.keras.layers.Resizing(
            height=layer_params['height'],
            width=layer_params['width']
        )
    elif layer_type == 'gdn':
        return gdn_func
    elif layer_type == 'prelu':
        return tf.keras.layers.PReLU(shared_axes=[1, 2])
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")