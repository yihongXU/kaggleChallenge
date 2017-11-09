from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import tensorflow as tf
import os


### parameters ###
images_dir = "/home/yxu/kaggle/test/"
graph = "/home/yxu/kaggle/tensorflow-for-poets-2/image_retraining/tf_files/retrained_graph_inception_4000_2w_distortion5per.pb"
labels_txt = "/home/yxu/kaggle/tensorflow-for-poets-2/image_retraining/tf_files/retrained_labels.txt"
output_layer = 'final_result:0'
input_layer = 'DecodeJpeg/contents:0'
in_csv = "/home/yxu/kaggle/sample_submission.csv"
out_csv = "/home/yxu/kaggle/submission_4000_2w_distortion5per .csv"
submission_sample_csv = "/home/yxu/kaggle/sample_submission.csv"

# to replace these names by the convension of sample_submission.csv
old_names = ['german_short_haired_pointer', 'shih_tzu', 'wire_haired_fox_terrier', 'black_and_tan_coonhound',
               'flat_coated_retriever', 'soft_coated_wheaten_terrier', 'curly_coated_retriever']
new_names = ['german_short-haired_pointer','shih-tzu', 'wire-haired_fox_terrier', 'black-and-tan_coonhound',
             'flat-coated_retriever', 'soft-coated_wheaten_terrier', 'curly-coated_retriever']
### prediction ###



def load_image(filename):
  """Read in the image_data to be classified."""
  return tf.gfile.FastGFile(filename, 'rb').read()


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def run_graph(images_dir, labels, input_layer_name, output_layer_name):
  with tf.Session() as sess:
    # Feed the image_data as input to the graph.
    #   predictions will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)

    results = []
    images_list = []
    #read image id from sample_submmsion.csv

    with open(submission_sample_csv, 'rb') as sample_f:
      reader = csv.reader(sample_f)
      for line_idx, row in enumerate(reader):  # row = list of format [id, breed]
        if line_idx == 0:  # first line is the header
          continue
        else:
          images_list.append(row[0]+'.jpg')
    print (images_list)
    sample_f.close()

    for img_name in images_list:
      if not tf.gfile.Exists(images_dir+img_name):
        tf.logging.fatal('image file does not exist %s', images_dir+img_name)
      image_data = load_image(images_dir+img_name)
      prediction, = sess.run(softmax_tensor, {input_layer_name: image_data})
      pred = {"id":img_name[:-4]}

      #  stock all probabilities for all breeds (label names)

      for node_id, confidence in enumerate(prediction):
        #  change label names to be consistent with submission.csv convention
        label_name = labels[node_id].replace(" ", "_")
        if label_name in old_names:
          label_name = new_names[old_names.index(label_name)]
        pred[label_name] = confidence
      results.append(pred)

    return results

def main(argv):
  """Runs inference on multiple images."""

  if not tf.gfile.Exists(labels_txt):
    tf.logging.fatal('labels file does not exist %s', labels_txt)

  if not tf.gfile.Exists(graph):
    tf.logging.fatal('graph file does not exist %s', graph)


  # load labels
  labels = load_labels(labels_txt)

  # load graph, which is stored in the default session
  load_graph(graph)

  # save result to dict, easy for write csv
  result = run_graph(images_dir, labels, input_layer, output_layer)

  # read header and write prediction csv
  headers = []
  with open(in_csv, 'rb') as in_f:
    reader = csv.reader(in_f)
    for line_idx, row in enumerate(reader):  # row = list of format [id, breed]
      headers = row
      break
  in_f.close()

  with open(out_csv, 'w') as out_f:
    f_csv = csv.DictWriter(out_f, headers)
    f_csv.writeheader()
    f_csv.writerows(result)
  out_f.close()


if __name__ == '__main__':
  tf.app.run(main=main)