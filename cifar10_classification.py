import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

(x, y), (x_test, y_test) = cifar10.load_data()
x, x_test = x/255.0, x_test/255.0
y = to_categorical(y, 10)
y_test = to_categorical(y_test, 10)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)


model = Sequential([
    Conv2D(64, (3,3), activation='relu', padding = 'same', input_shape = (32, 32, 3)),
    BatchNormalization(),
    Conv2D(64,(3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    Dropout(0.2),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    Flatten(),

    Dense(512, activation = 'relu'),
    Dropout(0.5),

    Dense(256, activation = 'relu'),
    Dropout(0.5),

    Dense(10, activation = 'softmax')
])

optimizer = Adam(learning_rate = 0.0005)
model.compile(optimizer = optimizer, loss='categorical_crossentropy', metrics = ['accuracy'])

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience=5, min_lr = 1e-6, verbose = 1)
datagen = ImageDataGenerator(
    rotation_range = 10, 
    width_shift_range = 0.05,
    height_shift_range = 0.05,
    horizontal_flip = True,
    zoom_range = 0.1
)

datagen.fit(x_train)

history = model.fit(datagen.flow(x_train, y_train, batch_size = 64), 
                    validation_data = (x_val, y_val), 
                    epochs =50,callbacks = [early_stopping, lr_scheduler] )

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")