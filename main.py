from model_ import get_ready_model
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle



NUM_EPOCHS = 2
BATCH_SIZE = 32

# load and preprocess dataset
train_dataset_path = r'data/training_images'
test_dataset_path = r'data/testing_images'

train_boxes_df = pd.read_csv(r'data/train_solution_bounding_boxes.csv')


def scale_bounding_boxes(coordinates: pd.DataFrame, nominal_shape: tuple, output_shape: tuple):
    """
    :param coordinates: with x1, y1, x2, y2 coordinates
    :param nominal_shape: (h, w) image shape
    :param output_shape: (h, w) image shape
    :return: array
    """
    h, w = nominal_shape
    new_h, new_w = output_shape

    coordinates.iloc[:, 0] = coordinates.iloc[:, 0] * (new_w/w)
    coordinates.iloc[:, 1] = coordinates.iloc[:, 1] * (new_h/h)
    coordinates.iloc[:, 2] = coordinates.iloc[:, 2] * (new_w/w)
    coordinates.iloc[:, 3] = coordinates.iloc[:, 3] * (new_h/h)
    return coordinates



def draw_box_filepath(file_path: str, box_coordinates: list):
    """Draw box on image from filepath"""
    x_min, y_min, x_max, y_max = box_coordinates
    width = x_max - x_min
    hight = y_max - y_min
    rectangle = Rectangle((x_min, y_min), width, hight, fill=False, color='white')
    image = plt.imread(file_path)
    plt.imshow(image)
    ax = plt.gca()
    ax.add_patch(rectangle)
    plt.show()

def draw_box_image(image, box_coordinates: list):
    x_min, y_min, x_max, y_max = box_coordinates
    width = x_max - x_min
    hight = y_max - y_min
    rectangle = Rectangle((x_min, y_min), width, hight, fill=False, color='white')
    plt.imshow(image)
    ax = plt.gca()
    ax.add_patch(rectangle)
    plt.show()

# show example image
example_path = r'data/training_images/vid_4_10060.jpg'
example_image = plt.imread(example_path)
print(example_image.shape)
plt.imshow(example_image)
# draw bounding box
filename = example_path.split(r'/')[-1]
coordinates = train_boxes_df[train_boxes_df['image'] == filename].values[0][1:]
for i in range(1):
    draw_box_filepath(example_path, coordinates)


images_reseized = tf.image.resize(example_image, (224, 224), method='nearest')


plt.imshow(images_reseized)
plt.figure(2)
plt.imshow(example_image)

coordinates2 = np.zeros((4, ))
coordinates2[0] = coordinates[0] * (224/676)
coordinates2[1] = coordinates[1] * (224/380)
coordinates2[2] = coordinates[2] * (224/676)
coordinates2[3] = coordinates[3] * (224/380)
plt.figure(3)
for i in range(1):
    draw_box_image(example_image, coordinates)

plt.figure(3)
for i in range(1):
    draw_box_image(images_reseized, coordinates2)



# build model
model = get_ready_model()
model.summary()



