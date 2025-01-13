#This function includes the custom loss functions.
# To add a new function, define here, add entry in config under loss_func, and entry under compile_model() function in train.py 

import tensorflow as tf
from tensorflow.keras.applications import VGG16

def sobel_edge_loss(y_true, y_pred):

    epsilon = 1e-7  # Small constant for numerical stability
    
    def sobel_edges(img):
        # Apply Sobel filters in x and y directions
        sobel_x = tf.image.sobel_edges(img)[..., 0]
        sobel_y = tf.image.sobel_edges(img)[..., 1]
        return tf.sqrt(tf.square(sobel_x) + tf.square(sobel_y) + epsilon)
    
    y_true_edges = sobel_edges(y_true)
    y_pred_edges = sobel_edges(y_pred)
    return tf.reduce_mean(tf.abs(y_true_edges - y_pred_edges))

def gradient_loss(y_true, y_pred):

    def compute_gradients(img):
        # Compute gradients in x and y directions using central differences
        gradient_x = img[:, :-1, 1:, :] - img[:, :-1, :-1, :]  # Horizontal gradient
        gradient_y = img[:, 1:, :-1, :] - img[:, :-1, :-1, :]  # Vertical gradient
        return gradient_x, gradient_y

    # Compute gradients for true and predicted images
    true_grad_x, true_grad_y = compute_gradients(y_true)
    pred_grad_x, pred_grad_y = compute_gradients(y_pred)

    # Compute the mean absolute error between gradients
    loss_x = tf.reduce_mean(tf.abs(true_grad_x - pred_grad_x))
    loss_y = tf.reduce_mean(tf.abs(true_grad_y - pred_grad_y))

    # Combine the gradient losses
    return loss_x + loss_y

def combined_loss(loss_functions, weights):
    def loss(y_true, y_pred):
        total_loss = 0
        for func, weight in zip(loss_functions, weights):
            total_loss += weight * func(y_true, y_pred)
        return total_loss
    return loss

class LossWeightScheduler(tf.keras.callbacks.Callback):
    def __init__(self, loss_weights, schedule):
        """
        Custom callback to adjust loss weights during training.
        :param loss_weights: List of tf.Variable objects corresponding to loss weights.
        :param schedule: A dictionary where keys are epoch numbers and values are lists of weights.
        """
        self.loss_weights = loss_weights
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.schedule:
            new_weights = self.schedule[epoch]
            for i, weight in enumerate(new_weights):
                self.loss_weights[i].assign(weight)
            print(f"Updated loss weights at epoch {epoch}: {new_weights}")

#This prints the relative contribution of each loss function, useful for debugging. Is functionally identical to combined_loss
def combined_loss_verbose(loss_functions, weights):

    def loss(y_true, y_pred):
        total_loss = 0
        loss_contributions = []  # To store contributions for printing

        for func, weight in zip(loss_functions, weights):
            # Compute the individual loss
            individual_loss = weight * func(y_true, y_pred)
            total_loss += individual_loss

            # Get the name of the function or class (fallback for custom functions)
            if hasattr(func, '__name__'):
                func_name = func.__name__
            elif hasattr(func, '__class__'):
                func_name = func.__class__.__name__
            else:
                func_name = "custom_loss"

            # Store loss contribution
            contribution = tf.reduce_mean(individual_loss)
            loss_contributions.append((func_name, contribution))
        
        # Print contributions (absolute values)
        tf.print("\nLoss Contributions (absolute):")
        for func_name, contribution in loss_contributions:
            tf.print(func_name, ":", contribution)

        return total_loss

    return loss

# Use a closure to create a perceptual loss function with the feature extractor
def perceptual_loss_with_extractor(y_true, y_pred):
    """
    Calculate perceptual loss by comparing features extracted from a VGG16 model.
    """
    # Resize input to match VGG input size
    y_true_resized = tf.image.resize(y_true, (224, 224))
    y_pred_resized = tf.image.resize(y_pred, (224, 224))

    # Extract features and compute mean squared error
    true_features = init_perceptual_loss.feature_extractor(y_true_resized)
    pred_features = init_perceptual_loss.feature_extractor(y_pred_resized)
    return tf.reduce_mean(tf.square(true_features - pred_features))

def init_perceptual_loss():
        # Initialize VGG16 model
        vgg = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        feature_extractor = tf.keras.Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv3').output)
        feature_extractor.trainable = False  # Ensure the feature extractor is not trainable
