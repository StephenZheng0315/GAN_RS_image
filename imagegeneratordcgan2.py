

# Remote Sensing Image Generation using GAN

##Dataset used: RSI-CB

## Install Libraries
"""

#!pip install pyunpack
#!pip install patool
url = 'https://public.sn.files.1drv.com/y4mYvqnmIYFMlKI9sD4Op_TbZRMbTRpQU0hdcorNRynCi-3MoUymHLC25SL2vrxSslUnDcclYq2Svt24w5xZFmwDHdgBrQ5YRG5ZlStBe3b3FA7aRMa95sOSzHeR53Quc_Y6cbVSM_7lYJT8rbFkBJdMDgLyI5VonfZXRaoSPwWf8p9XAlET8vci8fN4gT8VxJOI8FyoMtE0DgymXZ175M43w/RSI-CB128.rar?access_token=EwD4Aq1DBAAUcSSzoTJJsy%2bXrnQXgAKO5cj4yc8AAdDqDp%2bttiSWdJAQiYLCrv4V2%2fP0chEEmx5pOplb7kr92xZAuvmafi3tOFbksIbNjVo16BRtR3Xr2lkP%2bwdVYZLaaAJpeBRhRb8EjH2lGEAOjnpxOBms8%2f9YkqSfFJl5lt05zabufrbrsQzwQWFILrooIVEUir3FBsbvMHtzNlnag59yRL5FLG6yz%2f8CEZxCvwTqoc99otP728VHz5XxDhuPKGXwNR8WnB0x4KFayB%2b%2fI28d9%2fyvwd4E9WPEu%2bNCKOQOiLg%2bEkJjmrxkEkqDL3M%2bKemRD1uPbtriA5wjqUeavNMPbcqAWMakjGeNANeeoA8RDDKll9wxlKAH9Cy9yzQDZgAACBnksjgL%2bWVEyAGHVEqzboeH%2b1q6xyP4vcJ%2fa%2ba0Q1vvGTllAOGG72x0uDuzcM1qChjyRQ%2f0wIofyGG%2f8ttn7%2fmUzYWFU7l%2fQH5LdLIpWnNISpDldndok2xfkiLzAICEePjZ0BLMUFAUFTkCEhfwhZWq%2bw7PsFuUpE02dKOeT4rPt8gXC6pKIT96nm6jhGpNifMldIoq9tfBSShmyaUk2BpwAAAJmbQ%2bISZ2nwayFxuTvpbZHBfCC7rgQSzguWUPtDJmJYUtfwDBb92geXvtrWL0SoXQmnX5q1x1QeBILNmJKJ%2fTkKNDtngiJU0%2bYOFl88CaSXunig0Orsp%2fps%2bbMdzlswkF38h6r7pM2hLmu1lWwimEPHRBr35fLKntaSbqSbpCjgMG0%2fDM8gzgKqgZmQmazaLXm%2bWppFWAZ0oLQwnEMYvsnykYRrwNoX1FchqMLJ4Vlhi%2bohgH2gkjW7IXY3Vns6G3iAHESXRwdLjSimD6n6ptrD07HSHXZZ9B3NPPqAwhR1OenWbe3VN4CMsO6%2fdkNMKyILBzMHA8vETiX%2b%2bL47rQwjZ3rAcISyxgmSSdZ9T7i2ufYPZyL6FkLDFv%2fCLQIUEVMH9HBecmyp%2bruRTBNwgHAg%3d%3d'

"""##Import Libraries"""

import os
import time
import tensorflow as tf
import numpy as np
from glob import glob
import datetime
import random
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from urllib.request import urlretrieve
from pyunpack import Archive


# %matplotlib inline

"""##Create Directories"""

path = '/content/'

if not os.path.exists(path + 'rsensing'):
    os.makedirs(path + 'rsensing')
    
if not os.path.exists(path + 'rsensing' + '/data'):
    os.makedirs(path + 'rsensing' + '/data')
    
if not os.path.exists(path + 'rsensing' + '/generated'):
    os.makedirs(path + 'rsensing' + '/generated')
    
INPUT_DATA_DIR = path + "rsensing/data/RSI-CB128/water area/hirst/" 
OUTPUT_DIR = path + 'rsensing/generated/'

"""## Download and Extract Dataset"""

# DOWNLOAD DATASET

data_dir = str(path) + 'rsensing/data/'

if not os.path.exists(data_dir + 'RSI-CB128.rar'):
  
  class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

  with DLProgress(unit='B', unit_scale=True, miniters=1, desc='RSI-CB Data Set') as pbar:
        urlretrieve(str(url), data_dir + 'RSI-CB128.rar', pbar.hook)

    
# EXTRACT DATASET    
if not os.path.exists(str(data_dir) + 'RSI-CB128'):
  Archive(str(data_dir) + 'RSI-CB128.rar').extractall(str(data_dir))

"""### Define Training Parameters"""

IMAGE_SIZE = 128
NOISE_SIZE = 100
BATCH_SIZE = 64
EPOCHS = 300
EPSILON = 0.00005
samples_num = 5
LR_P = [0.00004, 0.0004]

"""## Generator

*   Input: random vector noise of size  100
*   Output: Generated RGB Image of shape 128 X 128 X 3
"""

def generator(z, output_channel_dim, training):
    with tf.variable_scope("generator", reuse= not training):
      
        WEIGHT_INIT_STDDEV = 0.02
        k_init = tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV)
        kernel = [5,5]
        strides = [2,2]
        
        
        # 8x8x1024        
        fully_connected = tf.layers.dense(z, 8*8*8*IMAGE_SIZE)
        fully_connected = tf.reshape(fully_connected, (-1, 8, 8, 8*IMAGE_SIZE))
        fully_connected = tf.nn.leaky_relu(fully_connected)

        
        # 8x8x1024 -> 16x16x512
        trans_conv1 = tf.layers.conv2d_transpose(fully_connected, 3*IMAGE_SIZE, kernel, strides, "SAME",
                                                                              kernel_initializer=k_init) 
        batch_trans_conv1 = tf.layers.batch_normalization(trans_conv1, training=training, epsilon=EPSILON)
        
        trans_conv1_out = tf.nn.leaky_relu(batch_trans_conv1)
        
        
        # 16x16x512 -> 32x32x256
        trans_conv2 = tf.layers.conv2d_transpose(trans_conv1_out,2*IMAGE_SIZE,kernel, strides,"SAME",
                                                                           kernel_initializer=k_init)             
        batch_trans_conv2 = tf.layers.batch_normalization(trans_conv2, training=training, epsilon=EPSILON)        
        
        trans_conv2_out = tf.nn.leaky_relu(batch_trans_conv2)
        
        
        
        # 32x32x256 -> 64x64x128
        trans_conv3 = tf.layers.conv2d_transpose(trans_conv2_out,IMAGE_SIZE,kernel, strides,"SAME",
                                                                         kernel_initializer=k_init)        
        batch_trans_conv3 = tf.layers.batch_normalization(trans_conv3, training=training, epsilon=EPSILON)
        
        trans_conv3_out = tf.nn.leaky_relu(batch_trans_conv3)
        
        
        # 64x64x128 -> 128x128x64
        trans_conv4 = tf.layers.conv2d_transpose(trans_conv3_out,int(IMAGE_SIZE/2),kernel, strides,"SAME",
                                                                                kernel_initializer=k_init)       
        batch_trans_conv4 = tf.layers.batch_normalization(trans_conv4, training=training, epsilon=EPSILON)
        
        trans_conv4_out = tf.nn.leaky_relu(batch_trans_conv4)
        
        
        # 128x128x64 -> 128x128x3
        logits = tf.layers.conv2d_transpose(trans_conv4_out,3,kernel,[1,1],"SAME",
                                                        kernel_initializer=k_init)
        
        out = tf.tanh(logits, name="out")

        return out

"""##Discriminator

*   Input: 128 X 128 X 3 RGB image
*   Output: It's probability of being real
"""

def discriminator(x, reuse):
    with tf.variable_scope("discriminator", reuse=reuse): 
        
        WEIGHT_INIT_STDDEV = 0.02
        k_init = tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV)
        kernel = [5,5]
        stride = [2,2]
        # 128*128*3 -> 64x64x64 
        
        
        conv1 = tf.layers.conv2d(x,int(IMAGE_SIZE/2),kernel, stride,"SAME", kernel_initializer=k_init)        
        batch_norm1 = tf.layers.batch_normalization(conv1, training=True, epsilon=EPSILON)        
        conv1_out = tf.nn.leaky_relu(batch_norm1)
        
        
        # 64x64x64-> 32x32x128 
        conv2 = tf.layers.conv2d(conv1_out,IMAGE_SIZE,kernel, stride,"SAME", kernel_initializer=k_init)       
        batch_norm2 = tf.layers.batch_normalization(conv2, training=True, epsilon=EPSILON)        
        conv2_out = tf.nn.leaky_relu(batch_norm2)
        
        
        # 32x32x128 -> 16x16x256  
        conv3 = tf.layers.conv2d(conv2_out,2*IMAGE_SIZE,kernel, stride,"SAME", kernel_initializer=k_init)        
        batch_norm3 = tf.layers.batch_normalization(conv3, training=True, epsilon=EPSILON)        
        conv3_out = tf.nn.leaky_relu(batch_norm3)
        
        
        # 16x16x256 -> 16x16x512
        conv4 = tf.layers.conv2d(conv3_out,3*IMAGE_SIZE,kernel,[1, 1],"SAME", kernel_initializer=k_init)        
        batch_norm4 = tf.layers.batch_normalization(conv4, training=True, epsilon=EPSILON)        
        conv4_out = tf.nn.leaky_relu(batch_norm4)
        
        
        # 16x16x512 -> 8x8x1024
        conv5 = tf.layers.conv2d(conv4_out,8*IMAGE_SIZE,kernel, stride,"SAME", kernel_initializer=k_init)        
        batch_norm5 = tf.layers.batch_normalization(conv5, training=True, epsilon=EPSILON)        
        conv5_out = tf.nn.leaky_relu(batch_norm5)

        
        flatten = tf.reshape(conv5_out, (-1, 8*8*8*IMAGE_SIZE))
        
        logits = tf.layers.dense(inputs=flatten, units=1, activation=None)
        
        out = tf.sigmoid(logits)
        
        return out, logits

"""## Calculate loss and optimize model"""

def model_loss(input_real, input_z, output_channel_dim):
    
    BETA1 = 0.5
    LR_D, LR_G = LR_P

    g_model = generator(input_z, output_channel_dim, True)

    noisy_input_real = input_real + tf.random_normal(shape=tf.shape(input_real), mean=0.0,
                                                     stddev=random.uniform(0.0, 0.1), dtype=tf.float32)
    
    d_model_real, d_logits_real = discriminator(noisy_input_real, reuse=False)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)
    
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                         labels=tf.ones_like(d_model_real)*random.uniform(0.9, 1.0)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                         labels=tf.zeros_like(d_model_fake)))
    d_loss = tf.reduce_mean(0.5 * (d_loss_real + d_loss_fake))
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                    labels=tf.ones_like(d_model_fake)))
    
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if var.name.startswith("generator")]
    d_vars = [var for var in t_vars if var.name.startswith("discriminator")]
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    gen_updates = [op for op in update_ops if op.name.startswith('generator')]
    
    with tf.control_dependencies(gen_updates):
        d_train_opt = tf.train.AdamOptimizer(learning_rate=LR_D, beta1=BETA1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate=LR_G, beta1=BETA1).minimize(g_loss, var_list=g_vars)  
    return d_loss, g_loss, d_train_opt, g_train_opt

"""## Create Placeholders"""

def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, *real_dim), name='inputs_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name="input_z")
    learning_rate_G = tf.placeholder(tf.float32, name="lr_g")
    learning_rate_D = tf.placeholder(tf.float32, name="lr_d")
    return inputs_real, inputs_z, learning_rate_G, learning_rate_D

"""## Display loss and sample images"""

def display_loss(epoch, time, sess, d_losses, g_losses, input_z, data_shape):
  
    minibatch_size = int(data_shape[0]//BATCH_SIZE)
    
    print("Epoch {}/{}".format(epoch, EPOCHS))
    print("Duration: ", round(time, 5))
    print("D_Loss: ", round(np.mean(d_losses[-minibatch_size:]), 5))
    print("G_Loss: ", round(np.mean(g_losses[-minibatch_size:]), 5))
          
    out_channel_dim = data_shape[3]
    fig, ax = plt.subplots()
    plt.plot(d_losses, label='Discriminator', alpha=0.6)
    plt.plot(g_losses, label='Generator', alpha=0.6)
    plt.title("Losses")
    plt.legend()
    plt.savefig(OUTPUT_DIR + "losses_" + str(epoch) + ".png")
    plt.show()
    plt.close()
    example_z = np.random.uniform(-1, 1, size=[samples_num, input_z.get_shape().as_list()[-1]])
    samples = sess.run(generator(input_z, out_channel_dim, False), feed_dict={input_z: example_z})
    sample_images = [((sample + 1.0) * 127.5).astype(np.uint8) for sample in samples]

    show_samples(sample_images, OUTPUT_DIR + "samples", epoch)

    
def show_samples(sample_images, name, epoch):
  for i in range(5):
    plt.imshow(sample_images[i])
    plt.show()
    img = Image.fromarray(np.uint8((sample_images[i]) * 255))
    img.save(str(OUTPUT_DIR) + str(epoch) + '_' + str(i) + '.png')

  plt.close()

"""## Create Batches"""

def get_batches(data):
    batches = []
    for i in range(int(data.shape[0]//BATCH_SIZE)):
        batch = data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        augmented_images = []
        for img in batch:
            image = Image.fromarray(img)
            if random.choice([True, False]):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            augmented_images.append(np.asarray(image))
        batch = np.asarray(augmented_images)
        normalized_batch = (batch / 127.5) - 1.0
        batches.append(normalized_batch)
    return batches

"""## Train Model"""

def train(get_batches, data_shape, checkpoint_to_load=None):
    input_images, input_z, lr_G, lr_D = model_inputs(data_shape[1:], NOISE_SIZE)
    d_loss, g_loss, d_opt, g_opt = model_loss(input_images, input_z, data_shape[3])
    
    LR_D, LR_G = LR_P
    saver = tf.train.Saver()

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        epoch = 0
        iteration = 0
        d_losses = []
        g_losses = []
        
        for epoch in range(EPOCHS):        
            epoch += 1
            t1 = time.time()

            for batch_images in get_batches:
                iteration += 1
                batch_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, NOISE_SIZE))
                _ = sess.run(d_opt, feed_dict={input_images: batch_images, input_z: batch_z, lr_D: LR_D})
                _ = sess.run(g_opt, feed_dict={input_images: batch_images, input_z: batch_z, lr_G: LR_G})

                d_losses.append(d_loss.eval({input_z: batch_z, input_images: batch_images}))
                g_losses.append(g_loss.eval({input_z: batch_z}))

            display_loss(epoch, time.time()-t1, sess, d_losses, g_losses, input_z, data_shape)
            
            #  S A V I N G  M O D E L        

            if epoch % 10 == 0 and epoch != 0:
              if not os.path.exists(path + 'rsensing/saved_model'):
                  os.makedirs(path + 'rsensing/saved_model')
              saver.save(sess, path + 'rsensing/saved_model' + '/model-' + str(epoch) + '.cptk')
              print ("Model Saved:", str(epoch) + '.cptk')
          
          
input_images = np.asarray([np.asarray(Image.open(file).resize((IMAGE_SIZE, IMAGE_SIZE))) for file in glob(INPUT_DATA_DIR + '*')])
print ("Input: " + str(input_images.shape))

np.random.shuffle(input_images)

sample_images = random.sample(list(input_images), samples_num)
show_samples(sample_images, OUTPUT_DIR + "inputs", 0)

with tf.Graph().as_default():
    train(get_batches(input_images), input_images.shape)
