import math

import tensorflow as tf

from models.channellayer import AWGNChannel, RayleighChannel, RicianChannel
from utils.patch_utils import build_blend_window, extract_overlapping_patches, fold_overlapping_patches


class FiLMLayer(tf.keras.layers.Layer):
    """Simple FiLM conditioning layer reused by the patch model."""

    def __init__(self, hidden_units=64, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_units, activation="relu"),
            tf.keras.layers.Dense(hidden_units, activation="relu"),
        ])
        self.gamma_layer = None
        self.beta_layer = None

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.gamma_layer = tf.keras.layers.Dense(channels)
        self.beta_layer = tf.keras.layers.Dense(channels)
        super().build(input_shape)

    def call(self, x, snr_db):
        if snr_db is None:
            return x

        snr_db = tf.cast(snr_db, tf.float32)
        if snr_db.shape.rank == 0:
            snr_db = tf.reshape(snr_db, [1, 1])
            snr_db = tf.repeat(snr_db, tf.shape(x)[0], axis=0)
        else:
            snr_db = tf.reshape(snr_db, [-1, 1])

        conditioning = self.mlp(snr_db)
        gamma = self.gamma_layer(conditioning)
        beta = self.beta_layer(conditioning)
        gamma = tf.reshape(gamma, [-1, 1, 1, tf.shape(x)[-1]])
        beta = tf.reshape(beta, [-1, 1, 1, tf.shape(x)[-1]])
        return x * (1.0 + gamma) + beta


class GroupNormalization(tf.keras.layers.Layer):
    """Minimal group normalization layer for batch-size-independent training."""

    def __init__(self, groups=8, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.groups = int(groups)
        self.epsilon = float(epsilon)
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        channels = int(input_shape[-1])
        group_count = min(self.groups, channels)
        while channels % group_count != 0 and group_count > 1:
            group_count -= 1
        self.groups = max(1, group_count)
        self.gamma = self.add_weight(name="gamma", shape=(channels,), initializer="ones", trainable=True)
        self.beta = self.add_weight(name="beta", shape=(channels,), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        channels_per_group = channels // self.groups

        x = tf.reshape(x, [batch_size, height, width, self.groups, channels_per_group])
        mean, variance = tf.nn.moments(x, axes=[1, 2, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(variance + self.epsilon)
        x = tf.reshape(x, [batch_size, height, width, channels])
        return x * self.gamma + self.beta


class ConvBlock(tf.keras.layers.Layer):
    """A small Conv-Norm-PReLU block used across the scalable patch architecture."""

    def __init__(
        self,
        filters,
        kernel_size=3,
        stride=1,
        normalization_type="batchnorm",
        group_norm_groups=8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.normalization_type = normalization_type
        use_norm = normalization_type != "none"
        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding="same",
            use_bias=not use_norm,
        )
        if normalization_type == "batchnorm":
            self.norm = tf.keras.layers.BatchNormalization()
        elif normalization_type == "groupnorm":
            self.norm = GroupNormalization(groups=group_norm_groups)
        else:
            self.norm = None
        self.activation = tf.keras.layers.PReLU(shared_axes=[1, 2])

    def call(self, x, training=False):
        x = self.conv(x)
        if self.norm is not None:
            if self.normalization_type == "batchnorm":
                x = self.norm(x, training=training)
            else:
                x = self.norm(x)
        return self.activation(x)


class ScalableEncoder(tf.keras.layers.Layer):
    """CNN encoder that preserves spatial flexibility and flattens to channel symbols."""

    def __init__(
        self,
        filters,
        latent_channels,
        downsample_factor,
        use_snr_side_info=False,
        film_hidden_units=64,
        normalization_type="batchnorm",
        group_norm_groups=8,
        name="scalable_encoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.filters = list(filters)
        self.latent_channels = int(latent_channels)
        self.downsample_factor = int(downsample_factor)
        self.use_snr_side_info = use_snr_side_info
        self.stem = ConvBlock(self.filters[0], kernel_size=5, stride=1, normalization_type=normalization_type, group_norm_groups=group_norm_groups, name=f"{name}_stem")
        self.blocks = []
        self.film_layers = []

        current_downsample = 1
        block_index = 0
        while current_downsample < self.downsample_factor:
            for filters_value in self.filters:
                stride = 2 if current_downsample < self.downsample_factor else 1
                block = ConvBlock(filters_value, kernel_size=3, stride=stride, normalization_type=normalization_type, group_norm_groups=group_norm_groups, name=f"{name}_block_{block_index}")
                self.blocks.append(block)
                self.film_layers.append(
                    FiLMLayer(hidden_units=film_hidden_units, name=f"{name}_film_{block_index}")
                    if use_snr_side_info else None
                )
                block_index += 1
                if stride == 2:
                    current_downsample *= 2
                if current_downsample >= self.downsample_factor:
                    break

        self.projection = tf.keras.layers.Conv2D(self.latent_channels, kernel_size=1, padding="same", name=f"{name}_proj")
        self.output_film = (
            FiLMLayer(hidden_units=film_hidden_units, name=f"{name}_film_out")
            if use_snr_side_info else None
        )

    def call(self, x, snr_db=None, training=False):
        x = self.stem(x, training=training)
        for block, film in zip(self.blocks, self.film_layers):
            x = block(x, training=training)
            if film is not None:
                x = film(x, snr_db)
        x = self.projection(x)
        if self.output_film is not None:
            x = self.output_film(x, snr_db)

        spatial_shape = tf.shape(x)[1:3]
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, 2])
        return x, spatial_shape


class ScalableDecoder(tf.keras.layers.Layer):
    """CNN decoder that reshapes channel symbols back to spatial tensors."""

    def __init__(
        self,
        filters,
        latent_channels,
        output_channels,
        upsample_scales,
        output_activation=None,
        use_snr_side_info=False,
        film_hidden_units=64,
        normalization_type="batchnorm",
        group_norm_groups=8,
        name="scalable_decoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.filters = list(filters)
        self.latent_channels = int(latent_channels)
        self.output_channels = int(output_channels)
        self.upsample_scales = list(upsample_scales)
        self.use_snr_side_info = use_snr_side_info

        self.pre_blocks = []
        self.pre_films = []
        for index, filters_value in enumerate(self.filters):
            self.pre_blocks.append(ConvBlock(filters_value, kernel_size=3, stride=1, normalization_type=normalization_type, group_norm_groups=group_norm_groups, name=f"{name}_pre_{index}"))
            self.pre_films.append(
                FiLMLayer(hidden_units=film_hidden_units, name=f"{name}_pre_film_{index}")
                if use_snr_side_info else None
            )

        self.post_blocks = []
        self.post_films = []
        for scale_index, _scale in enumerate(self.upsample_scales):
            filters_value = self.filters[min(scale_index, len(self.filters) - 1)]
            self.post_blocks.append(ConvBlock(filters_value, kernel_size=3, stride=1, normalization_type=normalization_type, group_norm_groups=group_norm_groups, name=f"{name}_post_{scale_index}"))
            self.post_films.append(
                FiLMLayer(hidden_units=film_hidden_units, name=f"{name}_post_film_{scale_index}")
                if use_snr_side_info else None
            )

        self.output_layer = tf.keras.layers.Conv2D(
            self.output_channels,
            kernel_size=3,
            padding="same",
            activation=output_activation,
            name=f"{name}_out",
        )

    def call(self, latent, spatial_shape, snr_db=None, target_size=None, training=False):
        batch_size = tf.shape(latent)[0]
        x = tf.reshape(latent, [batch_size, spatial_shape[0], spatial_shape[1], self.latent_channels])

        for block, film in zip(self.pre_blocks, self.pre_films):
            x = block(x, training=training)
            if film is not None:
                x = film(x, snr_db)

        for scale, block, film in zip(self.upsample_scales, self.post_blocks, self.post_films):
            current_size = tf.shape(x)[1:3] * scale
            x = tf.image.resize(x, current_size, method="bilinear")
            x = block(x, training=training)
            if film is not None:
                x = film(x, snr_db)

        if target_size is not None:
            x = tf.image.resize(x, target_size, method="bilinear")
        return self.output_layer(x)


class RefinementNetwork(tf.keras.layers.Layer):
    """Small residual fusion network used after global/local feature fusion."""

    def __init__(self, hidden_filters=64, output_channels=3, depth=3, normalization_type="batchnorm", group_norm_groups=8, name="refinement_network", **kwargs):
        super().__init__(name=name, **kwargs)
        self.blocks = [
            ConvBlock(hidden_filters, kernel_size=3, stride=1, normalization_type=normalization_type, group_norm_groups=group_norm_groups, name=f"{name}_block_{index}")
            for index in range(depth)
        ]
        self.output_layer = tf.keras.layers.Conv2D(output_channels, kernel_size=3, padding="same", activation="sigmoid")

    def call(self, x, training=False):
        residual = x
        for block in self.blocks:
            residual = block(residual, training=training)
        return self.output_layer(residual)


class PatchDeepJSCC(tf.keras.Model):
    """Hierarchical DeepJSCC model with a global branch and overlapping local patches."""

    def __init__(
        self,
        global_downsample_size=256,
        patch_size=256,
        patch_stride=192,
        global_latent_channels=64,
        local_latent_channels=32,
        global_branch_filters=(64, 96, 128),
        local_branch_filters=(48, 64, 96),
        global_branch_output="rgb",
        local_branch_output="rgb",
        enable_global_branch=True,
        enable_local_branch=True,
        enable_refinement=True,
        refinement_filters=64,
        refinement_depth=3,
        overlap_power=2.0,
        normalization_type="batchnorm",
        group_norm_groups=8,
        has_gdn=True,
        num_symbols=512,
        snrdB=10,
        channel="AWGN",
        input_size=64,
        set_channel_filters=True,
        channel_filters=32,
        debug_file=None,
        use_snr_side_info=False,
        film_hidden_units=64,
        rician_k_factor=2.0,
        **kwargs,
    ):
        del has_gdn, num_symbols, input_size, set_channel_filters, channel_filters, debug_file
        super().__init__(**kwargs)

        self.global_downsample_size = int(global_downsample_size)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.global_latent_channels = int(global_latent_channels)
        self.local_latent_channels = int(local_latent_channels)
        self.global_branch_output = global_branch_output
        self.local_branch_output = local_branch_output
        self.enable_global_branch = bool(enable_global_branch)
        self.enable_local_branch = bool(enable_local_branch)
        self.enable_refinement = bool(enable_refinement)
        self.use_snr_side_info = bool(use_snr_side_info)
        self.default_snrdB = float(snrdB)
        self.rician_k_factor = float(rician_k_factor)
        self.normalization_type = normalization_type
        self.group_norm_groups = int(group_norm_groups)

        if self.patch_stride > self.patch_size:
            raise ValueError("patch_stride must be <= patch_size to avoid uncovered regions.")
        if not self.enable_global_branch and not self.enable_local_branch:
            raise ValueError("At least one branch must be enabled.")

        global_output_channels = 3 if global_branch_output == "rgb" else refinement_filters
        local_output_channels = 3 if local_branch_output == "rgb" else refinement_filters

        self.global_encoder = None
        self.global_decoder = None
        if self.enable_global_branch:
            global_downsample_factor = max(1, self.global_downsample_size // 32)
            global_num_upsamples = int(math.ceil(math.log2(global_downsample_factor))) if global_downsample_factor > 1 else 0
            self.global_encoder = ScalableEncoder(
                filters=global_branch_filters,
                latent_channels=self.global_latent_channels,
                downsample_factor=max(1, 2 ** global_num_upsamples),
                use_snr_side_info=self.use_snr_side_info,
                film_hidden_units=film_hidden_units,
                normalization_type=self.normalization_type,
                group_norm_groups=self.group_norm_groups,
                name="global_encoder",
            )
            self.global_decoder = ScalableDecoder(
                filters=list(reversed(global_branch_filters)),
                latent_channels=self.global_latent_channels,
                output_channels=global_output_channels,
                upsample_scales=[2] * global_num_upsamples,
                output_activation="sigmoid" if global_branch_output == "rgb" else None,
                use_snr_side_info=self.use_snr_side_info,
                film_hidden_units=film_hidden_units,
                normalization_type=self.normalization_type,
                group_norm_groups=self.group_norm_groups,
                name="global_decoder",
            )

        self.local_encoder = None
        self.local_decoder = None
        if self.enable_local_branch:
            local_downsample_factor = max(1, self.patch_size // 32)
            local_num_upsamples = int(math.ceil(math.log2(local_downsample_factor))) if local_downsample_factor > 1 else 0
            self.local_encoder = ScalableEncoder(
                filters=local_branch_filters,
                latent_channels=self.local_latent_channels,
                downsample_factor=max(1, 2 ** local_num_upsamples),
                use_snr_side_info=self.use_snr_side_info,
                film_hidden_units=film_hidden_units,
                normalization_type=self.normalization_type,
                group_norm_groups=self.group_norm_groups,
                name="local_encoder",
            )
            self.local_decoder = ScalableDecoder(
                filters=list(reversed(local_branch_filters)),
                latent_channels=self.local_latent_channels,
                output_channels=local_output_channels,
                upsample_scales=[2] * local_num_upsamples,
                output_activation="sigmoid" if local_branch_output == "rgb" else None,
                use_snr_side_info=self.use_snr_side_info,
                film_hidden_units=film_hidden_units,
                normalization_type=self.normalization_type,
                group_norm_groups=self.group_norm_groups,
                name="local_decoder",
            )
            self.local_window = build_blend_window(self.patch_size, local_output_channels, power=overlap_power)
        else:
            self.local_window = None

        self.refinement_network = (
            RefinementNetwork(
                hidden_filters=refinement_filters,
                output_channels=3,
                depth=refinement_depth,
                normalization_type=self.normalization_type,
                group_norm_groups=self.group_norm_groups,
            )
            if self.enable_refinement else None
        )
        self.direct_fusion = tf.keras.layers.Conv2D(3, kernel_size=1, padding="same", activation="sigmoid")
        self.global_rgb_adapter = tf.keras.layers.Conv2D(3, kernel_size=1, padding="same", activation="sigmoid")
        self.local_rgb_adapter = tf.keras.layers.Conv2D(3, kernel_size=1, padding="same", activation="sigmoid")
        self.fusion_adapter = tf.keras.layers.Conv2D(refinement_filters, kernel_size=1, padding="same")

        if channel == "Rayleigh":
            self.channel = RayleighChannel(snrdB)
        elif channel == "AWGN":
            self.channel = AWGNChannel(snrdB)
        elif channel == "Rician":
            self.channel = RicianChannel(snrdB, k=self.rician_k_factor)
        else:
            self.channel = tf.identity

        self.last_outputs = None

    def _apply_channel(self, latent, snr_db):
        if callable(getattr(self.channel, "call", None)) and self.channel is not tf.identity:
            return self.channel(latent, snrdB=snr_db)
        return self.channel(latent)

    def _resolve_snr(self, x):
        snr_db = None
        if isinstance(x, (tuple, list)):
            x, snr_db = x
        elif self.use_snr_side_info:
            snr_db = tf.fill([tf.shape(x)[0]], tf.cast(self.default_snrdB, tf.float32))
        return x, snr_db

    def _decode_global(self, images, snr_db, training=False):
        if not self.enable_global_branch:
            return None, None, 0

        target_size = tf.constant([self.global_downsample_size, self.global_downsample_size], dtype=tf.int32)
        global_input = tf.image.resize(images, target_size, method="area")
        global_latent, global_spatial_shape = self.global_encoder(global_input, snr_db=snr_db, training=training)
        noisy_global_latent = self._apply_channel(global_latent, snr_db)
        global_decoded = self.global_decoder(
            noisy_global_latent,
            global_spatial_shape,
            snr_db=snr_db,
            target_size=target_size,
            training=training,
        )
        resized_global = tf.image.resize(global_decoded, tf.shape(images)[1:3], method="bilinear")
        return resized_global, global_decoded, int(global_latent.shape[1]) if global_latent.shape[1] is not None else 0

    def _decode_local(self, images, snr_db, training=False):
        if not self.enable_local_branch:
            return None, 0

        patches, metadata = extract_overlapping_patches(images, self.patch_size, self.patch_stride)
        local_snr = None
        patches_per_image = metadata["patch_rows"] * metadata["patch_cols"]
        if snr_db is not None:
            local_snr = tf.repeat(snr_db, repeats=patches_per_image)

        local_latent, local_spatial_shape = self.local_encoder(patches, snr_db=local_snr, training=training)
        noisy_local_latent = self._apply_channel(local_latent, local_snr)
        local_decoded_patches = self.local_decoder(
            noisy_local_latent,
            local_spatial_shape,
            snr_db=local_snr,
            target_size=tf.constant([self.patch_size, self.patch_size], dtype=tf.int32),
            training=training,
        )

        output_channels = int(local_decoded_patches.shape[-1]) if local_decoded_patches.shape[-1] is not None else 3
        output_shape = (
            tf.shape(images)[0],
            metadata["image_height"],
            metadata["image_width"],
            output_channels,
        )
        local_canvas = fold_overlapping_patches(
            local_decoded_patches,
            metadata,
            output_shape=output_shape,
            stride=self.patch_stride,
            window=self.local_window,
        )
        latent_uses = int(local_latent.shape[1]) if local_latent.shape[1] is not None else 0
        return local_canvas, latent_uses * patches_per_image

    def reconstruct_with_details(self, inputs, training=False):
        images, snr_db = self._resolve_snr(inputs)
        images = tf.convert_to_tensor(images, dtype=tf.float32)

        global_full_res, global_low_res, global_channel_uses = self._decode_global(images, snr_db, training=training)
        local_full_res, local_channel_uses = self._decode_local(images, snr_db, training=training)

        fusion_inputs = []
        if global_full_res is not None:
            fusion_inputs.append(global_full_res)
        if local_full_res is not None:
            fusion_inputs.append(local_full_res)
        fused_input = tf.concat(fusion_inputs, axis=-1)

        if self.refinement_network is not None:
            refined_features = self.fusion_adapter(fused_input)
            final_reconstruction = self.refinement_network(refined_features, training=training)
        else:
            final_reconstruction = self.direct_fusion(fused_input)

        details = {
            "input": images,
            "global_reconstruction": self.global_rgb_adapter(global_full_res) if global_full_res is not None and self.global_branch_output != "rgb" else global_full_res,
            "global_low_res_reconstruction": global_low_res,
            "local_reconstruction": self.local_rgb_adapter(local_full_res) if local_full_res is not None and self.local_branch_output != "rgb" else local_full_res,
            "final_reconstruction": final_reconstruction,
            "channel_uses": int(global_channel_uses) + int(local_channel_uses),
        }
        self.last_outputs = details
        return final_reconstruction, details

    def call(self, inputs, training=False):
        reconstruction, _ = self.reconstruct_with_details(inputs, training=training)
        return reconstruction
