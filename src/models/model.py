from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Activation, Input
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam
from src.constants.constants import *
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import tensorflow as tf
from src import logger
import numpy as np


class ImageClassifier:
    """
    A deep learning image classifier based on the Xception architecture.

    This class implements a transfer learning approach for image classification
    using the Xception model as a feature extractor. It provides a complete
    pipeline for model building, training, evaluation, and inference.

    The architecture consists of:
    1. Xception base model (pre-trained on ImageNet by default)
    2. Global average pooling to reduce feature dimensions
    3. Dense layer with configurable units and activation
    4. Output layer with softmax activation for multi-class classification

    Features:
    - Configurable learning rate and model architecture
    - Training with validation split
    - Model evaluation and visualization
    - Prediction of class probabilities and labels
    - Model persistence (save/load)
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
        num_classes: int = NUM_CLASSES,
        learning_rate: float = LEARNING_RATE,
        base_model_trainable: bool = BASE_MODEL_TRAINABLE,
        base_model_weights: str = BASE_MODEL_WEIGHTS,
    ) -> None:
        """
        Initialize the image classifier with the specified configuration.

        Args:
            input_shape: Shape of input images as (height, width, channels).
                        Default is defined in constants.
            num_classes: Number of target classification categories.
                        Default is defined in constants.
            learning_rate: Initial learning rate for the Adam optimizer.
                          Default is defined in constants.
            base_model_trainable: Whether the base Xception model layers should be trainable.
                                If False (default), the pre-trained weights will be frozen.
            base_model_weights: Pre-trained weights to use. Options are 'imagenet' (default)
                              for ImageNet pre-trained weights or None for random initialization.

        Raises:
            ValueError: If input dimensions are incompatible with the Xception model
            ImportError: If TensorFlow or required dependencies are not properly installed
        """
        try:
            self.learning_rate = learning_rate
            self.input_shape = input_shape
            self.num_classes = num_classes
            self.base_model_trainable = base_model_trainable

            logger.info(f"Initializing ImageClassifier with {num_classes} classes")
            logger.info(f"Input shape: {input_shape}, Learning rate: {learning_rate}")

            self.base_model = Xception(
                include_top=False,
                weights=base_model_weights,
                input_shape=self.input_shape,
            )
            logger.info(
                f"Loaded Xception base model with weights: {base_model_weights}"
            )

            self.model = self.build_model()
            logger.info("Model architecture successfully built")

        except Exception as e:
            logger.error(f"Error initializing ImageClassifier: {str(e)}")
            raise

    def build_model(self) -> tf.keras.Model:
        """
        Construct the neural network architecture.

        Creates a model that consists of the Xception base model followed by
        global average pooling and dense layers for classification.

        The architecture follows this structure:
        1. Input layer matching the specified input_shape
        2. Xception base model (with trainable status as specified)
        3. Global Average Pooling to reduce spatial dimensions
        4. Dense hidden layer with units and activation from constants
        5. Output layer with softmax activation for classification

        Returns:
            A Keras model ready for compilation and training

        Raises:
            RuntimeError: If model construction fails
        """
        try:
            self.base_model.trainable = self.base_model_trainable
            logger.info(f"Set base model trainable to: {self.base_model_trainable}")

            inputs = Input(shape=self.input_shape)
            x = self.base_model(inputs)
            x = GlobalAveragePooling2D()(x)
            x = Dense(DENSE_UNITS)(x)
            x = Activation(DENSE_ACTIVATION)(x)

            outputs = Dense(self.num_classes, activation=OUTPUT_ACTIVATION)(x)

            model = tf.keras.Model(inputs, outputs)
            logger.info(f"Model built with {model.count_params()} parameters")
            return model

        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise RuntimeError(f"Failed to build model: {str(e)}")

    def compile_model(self) -> None:
        """
        Compile the model with appropriate optimizer, loss function, and metrics.

        This method configures the model for training by specifying:
        - Adam optimizer with the configured learning rate
        - Loss function (typically categorical or sparse categorical cross-entropy)
        - Evaluation metrics (typically accuracy)

        All compilation parameters are defined in the constants file.

        This method must be called before training the model.

        Raises:
            RuntimeError: If model compilation fails
        """
        try:
            self.model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss=LOSS,
                metrics=METRICS,
            )
            logger.info(f"Model compiled with loss: {LOSS} and metrics: {METRICS}")
        except Exception as e:
            logger.error(f"Error compiling model: {str(e)}")
            raise RuntimeError(f"Failed to compile model: {str(e)}")

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        validation_split: float = VALIDATION_SPLIT,
    ) -> tf.keras.callbacks.History:
        """
        Train the model on the provided dataset.

        This method fits the model to the training data for the specified number
        of epochs, using the given batch size and validation split.

        Args:
            X_train: Training image data as a numpy array.
                    Should have shape (n_samples, height, width, channels).
            y_train: Training labels as a numpy array.
                    Should have shape (n_samples,) for sparse categorical labels.
            epochs: Number of complete passes through the training dataset.
                  Default is defined in constants.
            batch_size: Number of samples per gradient update.
                      Default is defined in constants.
            validation_split: Fraction of training data to use for validation.
                            Must be between 0 and 1. Default is defined in constants.

        Returns:
            A History object containing training metrics for each epoch

        Raises:
            ValueError: If input data dimensions are incorrect or validation_split is invalid
            RuntimeError: If training fails for any other reason
        """
        try:
            logger.info(f"Starting model training with {len(X_train)} samples")
            logger.info(
                f"Training parameters: epochs={epochs}, batch_size={batch_size}, validation_split={validation_split}"
            )

            history = self.model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
            )

            logger.info(
                f"Training completed. Final accuracy: {history.history['accuracy'][-1]:.4f}"
            )
            if "val_accuracy" in history.history:
                logger.info(
                    f"Validation accuracy: {history.history['val_accuracy'][-1]:.4f}"
                )

            return history

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise RuntimeError(f"Training failed: {str(e)}")

    def evaluate_model(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the model performance on test data.

        This method assesses how well the trained model performs on unseen data
        by computing the loss and accuracy metrics.

        Args:
            X_test: Test image data as a numpy array.
                  Should have shape (n_samples, height, width, channels).
            y_test: Test labels as a numpy array.
                  Should have shape (n_samples,) for sparse categorical labels.

        Returns:
            Dictionary containing evaluation metrics:
            - 'loss': The value of the model's loss function on test data
            - 'accuracy': The classification accuracy on test data

            Both metrics are rounded to 3 decimal places.

        Raises:
            ValueError: If input data dimensions are incorrect
            RuntimeError: If evaluation fails for any other reason
        """
        try:
            logger.info(f"Evaluating model on {len(X_test)} test samples")

            results = self.model.evaluate(X_test, y_test)

            metrics = {
                "loss": round(results[0], 3),
                "accuracy": round(results[1], 3),
            }

            logger.info(
                f"Evaluation results: loss={metrics['loss']}, accuracy={metrics['accuracy']}"
            )
            return metrics

        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise RuntimeError(f"Evaluation failed: {str(e)}")

    def visualize_training_history(
        self, history: tf.keras.callbacks.History, figsize: Tuple[int, int] = (12, 5)
    ) -> None:
        """
        Visualize the training and validation metrics over epochs.

        This method creates a figure with two subplots:
        1. Model accuracy (training and validation) over epochs
        2. Model loss (training and validation) over epochs

        The visualization helps in diagnosing overfitting, underfitting,
        and determining the optimal number of training epochs.

        Args:
            history: History object returned by model.fit() containing
                    training and validation metrics for each epoch.
            figsize: Size of the figure to display as (width, height) in inches.
                    Default is (12, 5).

        Raises:
            ValueError: If history object doesn't contain required metrics
            RuntimeError: If visualization fails for any other reason
        """
        try:
            logger.info("Generating training history visualization")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

            # Plot accuracy
            ax1.plot(
                history.history["accuracy"], label="Train", color="#1f77b4", linewidth=2
            )
            ax1.plot(
                history.history["val_accuracy"],
                label="Validation",
                color="#ff7f0e",
                linestyle="--",
            )
            ax1.set_title("Model Accuracy")
            ax1.set_ylabel("Accuracy")
            ax1.set_xlabel("Epoch")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Plot loss
            ax2.plot(
                history.history["loss"], label="Train", color="#2ca02c", linewidth=2
            )
            ax2.plot(
                history.history["val_loss"],
                label="Validation",
                color="#d62728",
                linestyle="--",
            )
            ax2.set_title("Loss")
            ax2.set_ylabel("Loss")
            ax2.set_xlabel("Epoch")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            plt.show()

            logger.info("Training history visualization displayed")

        except KeyError as e:
            logger.error(f"Missing key in history object: {str(e)}")
            raise ValueError(f"History object missing required metric: {str(e)}")
        except Exception as e:
            logger.error(f"Error visualizing training history: {str(e)}")
            raise RuntimeError(f"Visualization failed: {str(e)}")

    def predict_probabilities(
        self, X: np.ndarray, batch_size: int = BATCH_SIZE
    ) -> np.ndarray:
        """
        Generate class probability predictions for the input data.

        This method runs inference on the provided images and returns
        the probability distribution across all classes for each image.

        Args:
            X: Input image data as a numpy array.
              Should have shape (n_samples, height, width, channels).
            batch_size: Number of samples per batch for prediction.
                      Default is defined in constants.

        Returns:
            Array of predicted probabilities for each class.
            Shape is (n_samples, num_classes).

        Raises:
            ValueError: If input data dimensions are incorrect
            RuntimeError: If prediction fails for any other reason
        """
        try:
            logger.info(f"Generating probability predictions for {len(X)} samples")

            predictions = self.model.predict(X, batch_size=batch_size)

            logger.info("Probability predictions generated successfully")
            return predictions

        except Exception as e:
            logger.error(f"Error generating probability predictions: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def predict_classes(
        self, X: np.ndarray, batch_size: int = BATCH_SIZE
    ) -> np.ndarray:
        """
        Generate class label predictions for the input data.

        This method runs inference on the provided images and returns
        the most likely class label for each image.

        Args:
            X: Input image data as a numpy array.
              Should have shape (n_samples, height, width, channels).
            batch_size: Number of samples per batch for prediction.
                      Default is defined in constants.

        Returns:
            Array of predicted class indices (integers).
            Shape is (n_samples,).

            For binary classification (num_classes=2), returns 0 or 1.
            For multi-class classification, returns the index of the class
            with the highest probability.

        Raises:
            ValueError: If input data dimensions are incorrect
            RuntimeError: If prediction fails for any other reason
        """
        try:
            logger.info(f"Generating class predictions for {len(X)} samples")

            predictions = self.predict_probabilities(X, batch_size=batch_size)

            if self.num_classes == 2:
                class_predictions = (predictions > 0.5).astype(int).flatten()
            else:
                class_predictions = np.argmax(predictions, axis=1)

            logger.info("Class predictions generated successfully")
            return class_predictions

        except Exception as e:
            logger.error(f"Error generating class predictions: {str(e)}")
            raise RuntimeError(f"Class prediction failed: {str(e)}")

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.

        Saves the complete model (architecture, weights, and optimizer state)
        to the specified location for later use.

        Args:
            filepath: Path where the model should be saved

        Raises:
            RuntimeError: If saving the model fails
        """
        try:
            logger.info(f"Saving model to {filepath}")
            self.model.save(filepath)
            logger.info(f"Model successfully saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise RuntimeError(f"Failed to save model: {str(e)}")

    def load_model(self, filepath: str) -> None:
        """
        Load a previously saved model from disk.

        Loads a model that was previously saved using the save_model method,
        replacing the current model instance.

        Args:
            filepath: Path from which to load the model

        Raises:
            RuntimeError: If loading the model fails
        """
        try:
            logger.info(f"Loading model from {filepath}")
            self.model = tf.keras.models.load_model(filepath)
            logger.info(f"Model successfully loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def get_model_summary(self) -> None:
        """
        Print a summary of the model architecture.

        Displays information about each layer in the model, including
        layer type, output shape, and parameter count.
        """
        try:
            logger.info("Generating model summary")
            self.model.summary()
        except Exception as e:
            logger.error(f"Error generating model summary: {str(e)}")
