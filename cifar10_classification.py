import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


(x, y), (x_test, y_test) = cifar10.load_data()
x, x_test = x/255.0, x_test/255.0
y = to_categorical(y, 10)
y_test = to_categorical(y_test, 10)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)


model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding = 'same', input_shape = (32, 32, 3)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.2),

    Conv2D(64, (3,3), activation='relu', padding = 'same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation = 'relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(10, activation = 'softmax')
])


model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 10, batch_size= 64 )

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")