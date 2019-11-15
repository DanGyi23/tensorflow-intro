import tensorflow as tf

x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

result = tf.multiply(x1,x2)

with tf.Session() as sess:
  output = sess.run(result)
  print(output)

def load_data(data_directory):
  directories = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
  labels = []
  images = []

  for d in directories:
    label_directory = os.path.join(data_directory, d)
    file_names = [os.path.join(label_directory, f) for f in os.listdir(label_directory) if f.endswith(".ppm")]
    for f in file_names:
      images.append(skimage.data.imread(f))
      labels.append(int(d))
  return images, labels

  ROOT_PATH = '/Users/dangyi/Documents/Projects/tensorflow-intro'
  train_data_directory = os.path.join(ROOT_PATH, "Training")
  test_data_directory = os.path.join(ROOT_PATH, "Testing")

  images, labels = load_data(train_data_directory)