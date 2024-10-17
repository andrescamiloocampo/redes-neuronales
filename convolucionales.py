import tensorflow as tf
import tensorflow_datasets as tfds
import math

data,metadata = tfds.load('mnist',as_supervised=True,with_info=True)

training_data,test_data = data['train'],data['test']

def normalize(images,tags):
    images = tf.cast(images,tf.float32)
    images /= 255
    return images,tags

training_data = training_data.map(normalize)
test_data = test_data.map(normalize)

model = tf.keras.Sequential({
    tf.keras.layers.Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),input_shape=(28,28,1),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.Dense(units=100,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
})

num_training_data = metadata.splits["train"].num_examples
BATCH_SIZE = 32

record = model.fit(
    training_data,
    epochs=60,
    steps_per_epoch=math.ceil(num_training_data/BATCH_SIZE)
)

model.save('numeros_regular.h5')
