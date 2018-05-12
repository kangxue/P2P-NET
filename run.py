import argparse
import subprocess
import tensorflow as tf
import numpy as np
from datetime import datetime
import json
import os
import sys
import datetime
import time
import collections

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import P2PNET
import ioUtil

# DEFAULT SETTINGS
parser = argparse.ArgumentParser()

parser.add_argument('--train_hdf5', default='training data file name(*.hdf5)' )
parser.add_argument('--test_hdf5', default='test data file name(*.hdf5)' )

parser.add_argument('--domain_A', default='skeleton', help='name of domain A')
parser.add_argument('--domain_B', default='surface',  help='name of domain B')

parser.add_argument('--mode', type=str, default='train', help='train or test')
parser.add_argument('--gpu', type=int, default=0, help='which GPU to use [default: 0]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 4]')
parser.add_argument('--epoch', type=int, default=200, help='number of epoches to run [default: 200]')
parser.add_argument('--decayEpoch',  type=int, default=50, help='steps(how many epoches) for decaying learning rate')


parser.add_argument("--densityWeight", type=float, default=1.0, help="density weight [default: 1.0]")
parser.add_argument("--regularWeight", type=float, default=0.1, help="regularization weight [default: 0.1]")
parser.add_argument("--nnk", type=int, default=8, help="density:  number of nearest neighbours [default: 8]")

parser.add_argument("--range_max", type=float, default=1.0, help="max length of point displacement[default: 1.0]")
parser.add_argument("--radiusScal", type=float, default=1.0, help="a constant for scaling radii in pointnet++ [default: 1.0]")
parser.add_argument("--noiseLength", type=int, default=32, help="length of point-wise noise vector [default: 32]")

parser.add_argument('--checkpoint', default=None, help='epoch_##.ckpt')

###  None  None  None
parser.add_argument('--point_num', type=int, default=None, help='do not set the argument')
parser.add_argument('--example_num', type=int, default=None, help='do not set the argument')
parser.add_argument('--output_dir', type=str,  default=None, help='do not set the argument')

FLAGS = parser.parse_args()

Train_examples = ioUtil.load_examples(FLAGS.train_hdf5, FLAGS.domain_A, FLAGS.domain_B, 'names')
Test_examples  = ioUtil.load_examples(FLAGS.test_hdf5, FLAGS.domain_A, FLAGS.domain_B, 'names')

FLAGS.point_num = Train_examples.pointsets_A.shape[1]
POINT_NUM = FLAGS.point_num

Example_NUM = Train_examples.pointsets_A.shape[0]
FLAGS.example_num = Example_NUM

TRAINING_EPOCHES = FLAGS.epoch

batch_size = FLAGS.batch_size

if Train_examples.pointsets_B.shape[1] != POINT_NUM \
    or Test_examples.pointsets_A.shape[1] != POINT_NUM \
    or Test_examples.pointsets_B.shape[1] != POINT_NUM :
    print( 'point number inconsistent in the data set.')
    exit()

########## create output folders
datapath, basefname = os.path.split( FLAGS.train_hdf5 )
output_dir = 'output_' + basefname[0:basefname.index('_')] + '_' + FLAGS.domain_A + '-' + FLAGS.domain_B ## + '_noise' + str(FLAGS.noiseLength) + '_dw' + str(FLAGS.densityWeight)+ '_rw' + str(FLAGS.regularWeight)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

MODEL_STORAGE_PATH = os.path.join(output_dir, 'trained_models')
if not os.path.exists(MODEL_STORAGE_PATH):
    os.mkdir(MODEL_STORAGE_PATH)

SUMMARIES_FOLDER = os.path.join(output_dir, 'summaries')
if not os.path.exists(SUMMARIES_FOLDER):
    os.mkdir(SUMMARIES_FOLDER)


########## Save test input
ioUtil.output_point_cloud_ply( Test_examples.pointsets_A,  Test_examples.names, output_dir, 'gt_'+FLAGS.domain_A)
ioUtil.output_point_cloud_ply( Test_examples.pointsets_B, Test_examples.names, output_dir, 'gt_'+FLAGS.domain_B)

# print arguments
for k, v in FLAGS._get_kwargs():
    print(k + ' = ' + str(v) )


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(FLAGS.gpu)):
            model = P2PNET.create_model(FLAGS)

        ########## Init and Configuration   ##########
        saver = tf.train.Saver( max_to_keep=5 )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        init = tf.global_variables_initializer()
        sess.run(init)

        # Restore variables from disk.
        Start_epoch_number = 1
        if FLAGS.checkpoint is not None:
            print('load checkpoint: ' + FLAGS.checkpoint  )
            saver.restore(sess, FLAGS.checkpoint )

            fname = os.path.basename( FLAGS.checkpoint )
            Start_epoch_number = int( fname[6:-5] )  +  1

            print( 'Start_epoch_number = ' + str(Start_epoch_number) )


        train_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/test')

        fcmd = open(os.path.join(output_dir, 'arguments.txt'), 'w')
        fcmd.write(str(FLAGS))
        fcmd.close()


        ########## Training one epoch  ##########

        def train_one_epoch(epoch_num):

            now = datetime.datetime.now()
            print(now.strftime("%Y-%m-%d %H:%M:%S"))
            start_time = time.time()

            is_training = True

            Train_examples_shuffled = ioUtil.shuffle_examples(Train_examples)

            pointsets_A = Train_examples_shuffled.pointsets_A
            pointsets_B = Train_examples_shuffled.pointsets_B
            names = Train_examples_shuffled.names

            num_data = pointsets_A.shape[0]
            num_batch = num_data // batch_size

            total_data_loss_A = 0.0
            total_shape_loss_A = 0.0
            total_density_loss_A = 0.0

            total_data_loss_B = 0.0
            total_shape_loss_B = 0.0
            total_density_loss_B = 0.0

            total_reg_loss = 0.0

            for j in range(num_batch):

                begidx = j * batch_size
                endidx = (j + 1) * batch_size

                feed_dict = {
                    model.pointSet_A_ph: pointsets_A[begidx: endidx, ...],
                    model.pointSet_B_ph: pointsets_B[begidx: endidx, ...],
                    model.is_training_ph: is_training,
                }

                fetches = {
                    "train": model.total_train,
                    "shapeLoss_A": model.shapeLoss_A,
                    "densityLoss_A": model.densityLoss_A,
                    "shapeLoss_B": model.shapeLoss_B,
                    "densityLoss_B": model.densityLoss_B,
                    "data_loss_A": model.data_loss_A,
                    "data_loss_B": model.data_loss_B,
                    "regul_loss": model.regul_loss,
                    "learning_rate": model.learning_rate,
                    "global_step": model.global_step,
                }


                results = sess.run(fetches, feed_dict=feed_dict)

                total_data_loss_A += results["data_loss_A"]
                total_shape_loss_A += results["shapeLoss_A"]
                total_density_loss_A += results["densityLoss_A"]

                total_data_loss_B += results["data_loss_B"]
                total_shape_loss_B += results["shapeLoss_B"]
                total_density_loss_B += results["densityLoss_B"]

                total_reg_loss += results["regul_loss"]


                if j % 50 == 0:
                    print('    ' + str(j) + '/' + str(num_batch) + ':    '  )
                    print('            data_loss_A = {:.4f},'.format(results["data_loss_A"] )  +  \
                          '   shape = {:.4f},'.format(results["shapeLoss_A"] )           + \
                          '   density = {:.4f}'.format(results["densityLoss_A"] )   )

                    print('            data_loss_B = {:.4f},'.format(results["data_loss_B"] )  + \
                          '   shape = {:.4f},'.format(results["shapeLoss_B"] )           + \
                          '   density = {:.4f}'.format(results["densityLoss_B"] )   )

                    print('            regul_loss = {:.4f}\n'.format(results["regul_loss"] ) )

                    print('            learning_rate = {:.6f}'.format(results["learning_rate"] )  )
                    print('            global_step = {0}'.format(results["global_step"] )  )


            total_data_loss_A      /= num_batch
            total_shape_loss_A  /= num_batch
            total_density_loss_A   /= num_batch
            total_data_loss_B      /= num_batch
            total_shape_loss_B  /= num_batch
            total_density_loss_B   /= num_batch
            total_reg_loss         /= num_batch

            # evaluate summaries
            training_sum = sess.run( model.training_sum_ops, \
                                    feed_dict={model.train_dataloss_A_ph: total_data_loss_A, \
                                               model.train_dataloss_B_ph: total_data_loss_B, \
                                               model.train_regul_ph: total_reg_loss,\
                                               })

            train_writer.add_summary(training_sum, epoch_num)

            print(  '\tData_loss_A = %.4f,' % total_data_loss_A    + \
                    '    shape = %.4f,' % total_shape_loss_A + \
                    '    density = %.4f' % total_density_loss_A )

            print(  '\tData_loss_B = %.4f,' % total_data_loss_B    + \
                    '    shape = %.4f,' % total_shape_loss_B + \
                    '    density = %.4f' % total_density_loss_B )

            print(  '\tReg_loss: %.4f\n' % total_reg_loss)


            elapsed_time = time.time() - start_time
            print( '\tply/sec:' + str( round(num_data/elapsed_time) ) )
            print( '\tduration of this epoch:' + str(round(elapsed_time/60) ) + ' min' )
            print( '\testimated finishing time:' + str(round(elapsed_time/60.0 * (TRAINING_EPOCHES-epoch_num-1)) ) + ' min' )

        ################## end  of train function #################### end  of train function ##########


        def eval_one_epoch(epoch_num, mustSavePly=False):
            is_training = False

            pointsets_A = Test_examples.pointsets_A
            pointsets_B = Test_examples.pointsets_B
            names = Test_examples.names

            num_data = pointsets_A.shape[0]
            num_batch = num_data // batch_size


            total_data_loss_A = 0.0
            total_shape_loss_A = 0.0
            total_density_loss_A = 0.0

            total_data_loss_B = 0.0
            total_shape_loss_B = 0.0
            total_density_loss_B = 0.0

            total_reg_loss = 0.0

            for j in range(num_batch):

                begidx = j * batch_size
                endidx = (j + 1) * batch_size

                feed_dict = {
                    model.pointSet_A_ph: pointsets_A[begidx: endidx, ...],
                    model.pointSet_B_ph: pointsets_B[begidx: endidx, ...],
                    model.is_training_ph: is_training,
                }

                fetches = {
                    "shapeLoss_A": model.shapeLoss_A,
                    "densityLoss_A": model.densityLoss_A,
                    "shapeLoss_B": model.shapeLoss_B,
                    "densityLoss_B": model.densityLoss_B,
                    "data_loss_A": model.data_loss_A,
                    "data_loss_B": model.data_loss_B,
                    "regul_loss": model.regul_loss,
                    "Predicted_A": model.Predicted_A,
                    "Predicted_B": model.Predicted_B,
                }


                results = sess.run(fetches, feed_dict=feed_dict)

                total_data_loss_A += results["data_loss_A"]
                total_shape_loss_A += results["shapeLoss_A"]
                total_density_loss_A += results["densityLoss_A"]

                total_data_loss_B += results["data_loss_B"]
                total_shape_loss_B += results["shapeLoss_B"]
                total_density_loss_B += results["densityLoss_B"]

                total_reg_loss += results["regul_loss"]


                # write test results
                if epoch_num  % 20 == 0  or  mustSavePly:

                    # save predicted point sets with 1 single feeding pass
                    nametosave = names[begidx: endidx, ...]
                    Predicted_A_xyz = np.squeeze(np.array(results["Predicted_A"]))
                    Predicted_B_xyz = np.squeeze(np.array(results["Predicted_B"]))

                    ioUtil.output_point_cloud_ply(Predicted_A_xyz, nametosave, output_dir,
                                                    'Ep' + str(epoch_num) + '_predicted_' + FLAGS.domain_A  + 'X1')
                    ioUtil.output_point_cloud_ply(Predicted_B_xyz, nametosave, output_dir,
                                                    'Ep' + str(epoch_num) + '_predicted_' + FLAGS.domain_B  + 'X1')

                    # save predicted point sets with 4 feeding passes
                    for i in range(3):
                       results = sess.run(fetches, feed_dict=feed_dict)
                       Predicted_A_xyz__ = np.squeeze(np.array(results["Predicted_A"]))
                       Predicted_B_xyz__ = np.squeeze(np.array(results["Predicted_B"]))
                       Predicted_A_xyz = np.concatenate((Predicted_A_xyz, Predicted_A_xyz__), axis=1)
                       Predicted_B_xyz = np.concatenate((Predicted_B_xyz, Predicted_B_xyz__), axis=1)

                    ioUtil.output_point_cloud_ply(Predicted_A_xyz, nametosave, output_dir,
                                                   'Ep' + str(epoch_num) + '_predicted_' + FLAGS.domain_A + 'X4')
                    ioUtil.output_point_cloud_ply(Predicted_B_xyz, nametosave, output_dir,
                                                   'Ep' + str(epoch_num) + '_predicted_' + FLAGS.domain_B + 'X4')

                    # save predicted point sets with 8 feeding passes
                    for i in range(4):
                       results = sess.run(fetches, feed_dict=feed_dict)
                       Predicted_A_xyz__ = np.squeeze(np.array(results["Predicted_A"]))
                       Predicted_B_xyz__ = np.squeeze(np.array(results["Predicted_B"]))
                       Predicted_A_xyz = np.concatenate((Predicted_A_xyz, Predicted_A_xyz__), axis=1)
                       Predicted_B_xyz = np.concatenate((Predicted_B_xyz, Predicted_B_xyz__), axis=1)

                    ioUtil.output_point_cloud_ply( Predicted_A_xyz, nametosave, output_dir,
                                                   'Ep' + str(epoch_num) + '_predicted_' + FLAGS.domain_A + 'X8')
                    ioUtil.output_point_cloud_ply( Predicted_B_xyz, nametosave, output_dir,
                                                   'Ep' + str(epoch_num) + '_predicted_' + FLAGS.domain_B + 'X8')



            total_data_loss_A      /= num_batch
            total_shape_loss_A  /= num_batch
            total_density_loss_A   /= num_batch
            total_data_loss_B      /= num_batch
            total_shape_loss_B  /= num_batch
            total_density_loss_B   /= num_batch
            total_reg_loss         /= num_batch

            # evaluate summaries
            testing_sum = sess.run( model.testing_sum_ops, \
                                    feed_dict={model.test_dataloss_A_ph: total_data_loss_A, \
                                               model.test_dataloss_B_ph: total_data_loss_B, \
                                               model.test_regul_ph: total_reg_loss,\
                                               })

            test_writer.add_summary(testing_sum, epoch_num)

            print('\tData_loss_A = %.4f,' % total_data_loss_A  + \
                  '    shape = %.4f,' % total_shape_loss_A + \
                  '    density = %.4f' % total_density_loss_A)

            print('\tData_loss_B = %.4f,' % total_data_loss_B + \
                  '    shape = %.4f,' % total_shape_loss_B + \
                  '    density = %.4f' % total_density_loss_B)

            print('\tReg_loss: %.4f\n' % total_reg_loss)

        ################## end  of test function #################### end  of test function ##########

        if not os.path.exists(MODEL_STORAGE_PATH):
            os.mkdir(MODEL_STORAGE_PATH)

        if FLAGS.mode=='train':
            for epoch in range(Start_epoch_number,  TRAINING_EPOCHES+1):

                print( '\n>>> Training for the epoch %d/%d ...' % (epoch, TRAINING_EPOCHES))
                train_one_epoch(epoch)

                if epoch % 20 == 0:

                    cp_filename = saver.save(sess, os.path.join(MODEL_STORAGE_PATH, 'epoch_' + str(epoch) + '.ckpt'))
                    print( 'Successfully store the checkpoint model into ' + cp_filename)

                    print('\n<<< Testing on the test dataset...')
                    eval_one_epoch(epoch, mustSavePly=True)

        else:

            print( '\n<<< Testing on the test dataset ...')
            eval_one_epoch(Start_epoch_number, mustSavePly=True)



if __name__ == '__main__':
    train()
