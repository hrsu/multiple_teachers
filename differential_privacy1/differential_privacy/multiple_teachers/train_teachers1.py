# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import deep_cnn
import input
import metrics
import time
import random
import matplotlib.pyplot as plt

import os

tf.flags.DEFINE_string('dataset', 'svhn', 'The name of the dataset to use')
tf.flags.DEFINE_integer('nb_labels', 10, 'Number of output classes')

tf.flags.DEFINE_string('data_dir','/tmp','Temporary storage')
tf.flags.DEFINE_string('train_dir','/tmp/train_dir',
                       'Where model ckpt are saved')

tf.flags.DEFINE_integer('max_steps', 1001, 'Number of training steps to run.')
tf.flags.DEFINE_integer('nb_teachers', 10, 'Teachers in the ensemble.')
tf.flags.DEFINE_integer('teacher_id', 0, 'ID of teacher being trained.')

tf.flags.DEFINE_boolean('deeper', False, 'Activate deeper CNN model')

FLAGS = tf.flags.FLAGS

#传入lists和rate对lists进行干扰
def disturb(lists, rate):
  # lists = [1, 2, 3, 4, 5, 6, 7, 8, 9]
  # rate = 0.5
  # 获取标签最大最小值
  minnum = min(lists)
  maxnum = max(lists)

  # 根据rate计算需要干扰的标签个数
  lens = len(lists)
  num = int(lens * rate)

  # 随机选取num个标签位置存入s
  s = []
  while (len(s) < num):
    x = random.randint(0, lens - 1)  # 取随机位置
    if x not in s:
      s.append(x)

  # print(s)
  print("干扰前: ", lists)

  # 从minnum-maxnum内随机选取值(与原来不同)对指定位置进行干扰
  i = 0
  while i < (len(s)):
    j = s[i]
    newn = random.randint(minnum, maxnum)  # 生成一个随机数
    if (newn != lists[j]):
      lists[j] = newn
      i += 1

  print("干扰后: ", lists)
  # 返回干扰后的list
  return lists

def train_teacher(dataset, nb_teachers, teacher_id):
  """
  This function trains a teacher (teacher id) among an ensemble of nb_teachers
  models for the dataset specified.
  :param dataset: string corresponding to dataset (svhn, cifar10)
  :param nb_teachers: total number of teachers in the ensemble
  :param teacher_id: id of the teacher being trained
  :return: True if everything went well
  """
  # If working directories do not exist, create them
  assert input.create_dir_if_needed(FLAGS.data_dir)
  assert input.create_dir_if_needed(FLAGS.train_dir)
  print("teacher {}:".format(teacher_id))
  # Load the dataset
  if dataset == 'svhn':
    train_data,train_labels,test_data,test_labels = input.ld_svhn(extended=True)
  elif dataset == 'cifar10':
    train_data, train_labels, test_data, test_labels = input.ld_cifar10()
  elif dataset == 'mnist':
    train_data, train_labels, test_data, test_labels = input.ld_mnist()
  else:
    print("Check value of dataset flag")
    return False

  path = os.path.abspath('.')

  path1 = path + '\\plts_nodisturb\\'

  # 对标签进行干扰
  import copy
  train_labels1 = copy.copy(train_labels)
  train_labels2 = disturb(train_labels, 0.1)
  disturb(test_labels, 0.1)
  #path1 = path + '\\plts_withdisturb\\'

  # Retrieve subset of data for this teacher
  #干扰前
  data, labels = input.partition_dataset(train_data,
                                         train_labels,
                                         nb_teachers,
                                         teacher_id)

  from pca import K_S
  import operator
  print(operator.eq(train_labels1, train_labels2))
  print("干扰前: ", K_S.tst_norm(train_labels1))
  print("干扰后: ", K_S.tst_norm(train_labels2))
  print(K_S.tst_samp(train_labels1, train_labels2))

  print("Length of training data: " + str(len(labels)))

  # Define teacher checkpoint filename and full path
  if FLAGS.deeper:
    filename = str(nb_teachers) + '_teachers_' + str(teacher_id) + '_deep.ckpt'
  else:
    filename = str(nb_teachers) + '_teachers_' + str(teacher_id) + '.ckpt'
  ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + filename

  # Perform teacher training
  losses =  deep_cnn.train(data, labels, ckpt_path)

  # Append final step value to checkpoint for evaluation
  ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps - 1)

  # Retrieve teacher probability estimates on the test data
  teacher_preds = deep_cnn.softmax_preds(test_data, ckpt_path_final)

  # Compute teacher accuracy
  precision = metrics.accuracy(teacher_preds, test_labels)
  print('Precision of teacher after training: ' + str(precision))
  print("each n step loss: ", losses)

  #x = list(range(1, len(losses)+1))
  #plt.plot(x, losses, 'bo-', markersize=20)
  #plt.savefig(path1 + 'loss' + str(teacher_id) + '.jpg')
  #plt.show()
  #print("x: ",x)
  #print("loss: ", losses)

  return True

def main(argv=None):  # pylint: disable=unused-argument
  FLAGS.dataset = 'svhn'
  #'svhn', 'cifar10', 'mnist'
  FLAGS.nb_teachers = 5
  for i in range(5):
    start_time = time.time()
    FLAGS.teacher_id=i
    assert train_teacher(FLAGS.dataset, FLAGS.nb_teachers, FLAGS.teacher_id)
    end_time = time.time()
    print("it cost {}s.".format(round(end_time - start_time, 3)))
    print()

if __name__ == '__main__':
  FLAGS.dataset = 'cifar10'
  # 'svhn', 'cifar10', 'mnist'
  FLAGS.nb_teachers = 10
  for i in range(10):
    start_time=time.time()
    FLAGS.teacher_id=i
    assert train_teacher(FLAGS.dataset, FLAGS.nb_teachers, FLAGS.teacher_id)
    end_time = time.time()
    print("it cost {}s.".format(round(end_time-start_time,3)))
    print()
