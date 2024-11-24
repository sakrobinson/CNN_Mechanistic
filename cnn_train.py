import tensorflow as tf
from tensorflow.keras import layers, models 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

def create_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model



# Normalize the images
X = X / 255.0

# Define the cross-validator
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cvscores = []
histories = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
    print(f'\nTraining on fold {fold+1}...')
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = create_cnn_model(input_shape=(64, 64, 3))
    
    history = model.fit(X_train, y_train, epochs=5, batch_size=32,
                        validation_data=(X_test, y_test), verbose=1)
    
    # Evaluate the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f"Fold {fold+1} Test Accuracy: {scores[1]*100:.2f}%")
    cvscores.append(scores[1] * 100)
    histories.append(history)
    
    # Classification report
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print(classification_report(y_test, y_pred, target_names=['Circle', 'Square']))
