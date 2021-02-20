import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Model


def feature_extractor(inputs):
    feature_extractor_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return feature_extractor_model(inputs)


def dense_layers(features):
    x = tf.keras.layers.GlobalAveragePooling2D()(features)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    return x


def bounding_box_layer(x):
    bounding_box_output = tf.keras.layers.Dense(4, name='bounding_box')(x)
    return bounding_box_output


def assembly_model(inputs) -> Model:
    features = feature_extractor(inputs)
    x = dense_layers(features)
    bounding_box_output = bounding_box_layer(x)

    model = tf.keras.Model(inputs=inputs, outputs=bounding_box_output)
    return model


def compile_model():
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    model = assembly_model(inputs)
    model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9), loss="mean_squared_error")
    return model


def get_ready_model():
    return compile_model()
