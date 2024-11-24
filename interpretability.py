# Mechanistic Interpretability Testing O_o


# Select an input image (correctly classified or misclassified)
sample_image = X_test[0]
sample_image_label = y_test[0]
sample_image = np.expand_dims(sample_image, axis=0)  # Add batch dimension

# Get the outputs of all layers
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name or 'dense' in layer.name]

# Create a new model that will return these outputs, given the model input
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# Get the activations
activations = activation_model.predict(sample_image)

# Visualize the activations
layer_names = [layer.name for layer in model.layers if 'conv' in layer.name or 'dense' in layer.name]

images_per_row = 8

for layer_name, layer_activation in zip(layer_names, activations):
    if len(layer_activation.shape) == 4:
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                # Normalize the image for visualization
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std() + 1e-5
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.axis('off')
        plt.show()
    else:
        # For Dense layers
        print(f"{layer_name} activation shape: {layer_activation.shape}")
        print("Activation values:", layer_activation)
        print("-" * 50)

## Visualizing Convolutional Filters

# Get the weights of the first convolutional layer
filters, biases = model.layers[0].get_weights()
print(f"Filters shape: {filters.shape}")

n_filters = filters.shape[-1]
fig, axes = plt.subplots(1, n_filters, figsize=(20, 5))
for i in range(n_filters):
    f = filters[:, :, :, i]
    f_min, f_max = f.min(), f.max()
    # Normalize the filter to [0, 1] range for visualization
    f = (f - f_min) / (f_max - f_min)
    axes[i].imshow(f[:, :, 0], cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f'Filter {i+1}')
plt.show()

## Saliency Maps and Grad-CAM

# might need tf-explain
# !pip install tf-explain

from tf_explain.core.grad_cam import GradCAM

# Use the last convolutional layer
conv_layer_name = [layer.name for layer in model.layers if 'conv' in layer.name][-1]
explainer = GradCAM()
data = ([sample_image], None)
grid = explainer.explain(data, model, class_index=sample_image_label, layer_name=conv_layer_name)

# Visualize
plt.imshow(grid)
plt.title('Grad-CAM')
plt.axis('off')
plt.show()

## Model Errors

# Identify misclassified samples
misclassified_indices = np.where(y_pred != y_test)[0]

# Visualize misclassified samples
for idx in misclassified_indices[:5]:  # Show first 5 misclassified samples
    plt.imshow(X_test[idx])
    plt.title(f'True: {"Circle" if y_test[idx]==0 else "Square"}, Predicted: {"Circle" if y_pred[idx]==0 else "Square"}')
    plt.axis('off')
    plt.show()

# Analyze patterns in misclassified images
print("Number of misclassified samples:", len(misclassified_indices))

## Using TensorBoard for Visualization

from tensorflow.keras.callbacks import TensorBoard

# Create a new model instance for TensorBoard visualization
model_tb = create_cnn_model(input_shape=(64, 64, 3))

# Define TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model with TensorBoard callback
history_tb = model_tb.fit(X_train, y_train, epochs=5, batch_size=32,
                          validation_data=(X_test, y_test),
                          callbacks=[tensorboard_callback], verbose=1)

# After training, run this in the terminal:
# tensorboard --logdir logs/fit

## Ablation Studies

# Modify the model architecture (e.g., remove one convolutional layer)
def create_cnn_model_modified(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name='conv1_mod'),
        layers.MaxPooling2D((2, 2), name='pool1_mod'),
        # Removed second convolutional layer
        layers.Flatten(name='flatten_mod'),
        layers.Dense(64, activation='relu', name='dense1_mod'),
        layers.Dense(1, activation='sigmoid', name='output_mod')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Create and train the modified model
model_mod = create_cnn_model_modified(input_shape=(64, 64, 3))
history_mod = model_mod.fit(X_train, y_train, epochs=5, batch_size=32,
                            validation_data=(X_test, y_test), verbose=1)

# Evaluate the modified model
scores_mod = model_mod.evaluate(X_test, y_test, verbose=0)
print(f"Modified Model Test Accuracy: {scores_mod[1]*100:.2f}%")

# Compare with the original model's performance
print(f"Original Model Test Accuracy: {scores[1]*100:.2f}%")
print(f"Difference in Accuracy: {scores[1]*100 - scores_mod[1]*100:.2f}%")

# Plot training & validation accuracy values for both models
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], label='Original Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Original Validation Accuracy')
plt.plot(history_mod.history['accuracy'], label='Modified Train Accuracy', linestyle='--')
plt.plot(history_mod.history['val_accuracy'], label='Modified Validation Accuracy', linestyle='--')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()

# Plot training & validation loss values for both models
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Original Train Loss')
plt.plot(history.history['val_loss'], label='Original Validation Loss')
plt.plot(history_mod.history['loss'], label='Modified Train Loss', linestyle='--')
plt.plot(history_mod.history['val_loss'], label='Modified Validation Loss', linestyle='--')
plt.title('Model Loss Comparison')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()
