import numpy as np
import pandas as pd
import os
import random
import gc
import tensorflow as tf
from keras.models import Model, Sequential
from keras import backend as K
from keras.layers import (Input, Dense, Dropout, Flatten, BatchNormalization,
                          Conv2D, MultiHeadAttention, concatenate, Multiply,
                          Lambda, Add)
from tensorflow.keras.optimizers import Adam
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, precision_recall_fscore_support,
                             precision_recall_curve, confusion_matrix)
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def calc_confusion_matrix(result, test_label, mode, learning_rate, batch_size, epochs):
    test_label = to_categorical(test_label, 2)

    true_label = np.argmax(test_label, axis=1)
    predicted_label = np.argmax(result, axis=1)

    n_classes = 2
    precision = dict()
    recall = dict()
    thres = dict()

    for i in range(n_classes):
        precision[i], recall[i], thres[i] = precision_recall_curve(test_label[:, i], result[:, i])

    print("Classification Report :")
    print(classification_report(true_label, predicted_label, digits=4))
    cr = classification_report(true_label, predicted_label, output_dict=True)
    
    
    cm = confusion_matrix(true_label, predicted_label)
    print("Confusion Matrix :")
    print(cm)
    
    return cr, precision, recall, thres

def calculate_avg_and_variance(reports):
   
    avg_precision = {}
    avg_recall = {}
    avg_f1_score = {}
    variance_precision = {}
    variance_recall = {}
    variance_f1_score = {}

    
    for report in reports:
        for label, metrics in report.items():
            if label.isdigit():
                label = int(label)
                if label not in avg_precision:
                    avg_precision[label] = []
                    avg_recall[label] = []
                    avg_f1_score[label] = []
                avg_precision[label].append(metrics['precision'])
                avg_recall[label].append(metrics['recall'])
                avg_f1_score[label].append(metrics['f1-score'])

    # 计算每个类别的平均值和方差
    for label in avg_precision.keys():
        variance_precision[label] = np.var(avg_precision[label])
        variance_recall[label] = np.var(avg_recall[label])
        variance_f1_score[label] = np.var(avg_f1_score[label])

    return avg_precision, avg_recall, avg_f1_score, variance_precision, variance_recall, variance_f1_score

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def create_model_img():
    model = Sequential()
    model.add(Conv2D(72, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Flatten())
    model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.4))
    return model

def cross_modal_attention(x, y):
    x = tf.expand_dims(x, axis=1)
    y = tf.expand_dims(y, axis=1)
    a1 = MultiHeadAttention(num_heads=4, key_dim=50)(x, y)
    a2 = MultiHeadAttention(num_heads=4, key_dim=50)(y, x)
    a1 = a1[:, 0, :]
    a2 = a2[:, 0, :]
    return concatenate([a1, a2])  

def self_attention(x):
    x = tf.expand_dims(x, axis=1)
    attention = MultiHeadAttention(num_heads=4, key_dim=50)(x, x)
    attention = attention[:, 0, :]
    return attention  # 50维

def cross_modal_attention_block(input_tensor, common_tensor, num_layers=1):
    output_tensor = input_tensor
    for _ in range(num_layers):
        output_tensor = cross_modal_attention(output_tensor, common_tensor)
        output_tensor = Dense(50, activation=None, kernel_regularizer=l2(0.01))(output_tensor)
    return output_tensor

def multi_modal_model(mode, train_img1, train_img2):
    in_img1 = Input(shape=(train_img1.shape[1], train_img1.shape[2], train_img1.shape[3]))
    in_img2 = Input(shape=(train_img2.shape[1], train_img2.shape[2], train_img2.shape[3]))

    dense_img1 = create_model_img()(in_img1)
    dense_img2 = create_model_img()(in_img2)

    # 计算权重
    dense_img11 = Dense(50, activation=None, kernel_regularizer=l2(0.01))(dense_img1)
    dense_img22 = Dense(50, activation=None, kernel_regularizer=l2(0.01))(dense_img2)
    img_w1 = Dense(1, activation=None, kernel_regularizer=l2(0.01))(dense_img11)
    img_w2 = Dense(1, activation=None, kernel_regularizer=l2(0.01))(dense_img22)
    exp_img_w1 = Lambda(lambda x: K.exp(x))(img_w1)
    exp_img_w2 = Lambda(lambda x: K.exp(x))(img_w2)

    # 
    img_sum_exp = Add()([exp_img_w1, exp_img_w2])
    img_w1_normalized = Lambda(lambda x: x[0] / x[1])([exp_img_w1, img_sum_exp])
    img_w2_normalized = Lambda(lambda x: x[0] / x[1])([exp_img_w2, img_sum_exp])

    common = Multiply()([img_w1_normalized, dense_img1])
    common = Add()([common, Multiply()([img_w2_normalized, dense_img2])])

    common = Dense(50, activation=None, kernel_regularizer=l2(0.01))(common)

    ########### Attention Layer ############
    if mode == 'MM_SA_BA':
            common=self_attention(common)
            v_c=cross_modal_attention( dense_img1, common)#强化后的白质和common
            v_c= Dense(50, activation=None, kernel_regularizer=l2(0.01))(v_c)
            v_c=cross_modal_attention(v_c, common)
            v_c= Dense(50, activation=None, kernel_regularizer=l2(0.01))(v_c)
            v_c=cross_modal_attention(v_c, common)
            t_c=cross_modal_attention(dense_img2, common)
            t_c= Dense(50, activation=None, kernel_regularizer=l2(0.01))(t_c)
            t_c=cross_modal_attention(t_c, common) 
            t_c= Dense(50, activation=None, kernel_regularizer=l2(0.01))(t_c)
            t_c=cross_modal_attention(t_c, common)
         
            merged=concatenate([v_c,t_c,common,dense_img1,dense_img2])
    elif mode == 'None':
        merged = concatenate([dense_img1, dense_img2])
    else:
        print("Mode must be one of 'MM_SA_BA' or 'None'.")
        return

    ########### Output Layer ############
    output = Dense(2, activation='softmax')(merged)

    model = Model([in_img1, in_img2], output)

    return model


