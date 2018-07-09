# -*- coding: utf-8 -*-
"""
Basketball hoop detection

Created on Wed Jul  4 21:02:34 2018

@author: Cathey Wang
"""

import pandas as pd
import numpy as np
import cv2
import random
import time

from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPooling2D, \
    Activation, LeakyReLU, BatchNormalization, add
from keras import optimizers, losses, metrics
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

from PIL import Image
from PIL.ImageDraw import Draw
from scipy.ndimage import zoom


"""
Declare global parameters
"""
def global_params():
    global L
    L = 256     # input image size


"""
Edge coordinates - bounding box params conversion
"""
def edge2xywh(edge):
    x1, y1, x2, y2 = edge
    return [(x1 + x2)/2/L, (y1 + y2)/2/L, (x2 - x1)/L, (y2 - y1)/L]
    
def xywh2edge(xywh):
    x, y, w, h = xywh
    return [int((x-w/2)*L), int((y-h/2)*L), int((x+w/2)*L), int((y+h/2)*L)]


"""
3rd party code: Hue shift
"""
def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv

def hsv_to_rgb(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')

def hue_shift(img, amount):
    hsv = rgb_to_hsv(img)
    hsv[..., 0] = (hsv[..., 0]+amount) % 1.0
    rgb = hsv_to_rgb(hsv)
    return rgb


"""
Data augmentation:
1. Horizontal flipping      x2
2. Cropping (retain hoop)   x4
3. Hue adjustment           x3
4. Gaussian noise           x2
"""
def augment_data(X, Y):
    N = Y.shape[0]
    X_aug = np.zeros((N*2, L,L,3))
    Y_aug = np.zeros((N*2, 4))
    for n in range(N):          # for each unique image
        X_orig = X[n,:,:,:]
        Y_orig = Y[n,:]
        for a in range(2):      # horizontal flipping, y flips
            if a == 0:
                x = np.copy(X_orig)
                y = np.copy(Y_orig)
            else:
                x = np.fliplr(X_orig)
                y = np.copy(Y_orig)
                y[0] = 1-y[0]
            X_aug[n*2+a,:,:,:] = x
            Y_aug[n*2+a,:] = y
#            edge = xywh2edge(y)
#            for b in range(4):  # cropping, need to observe y
#                if b > 0:   # generate 3 of these
#                    r = random.uniform(0.75, 0.9)
#                    p = int(random.uniform(max(0, edge[2]-int(r*L)), min(edge[0], int((1-r)*L))))
#                    q = int(random.uniform(max(0, edge[3]-int(r*L)), min(edge[1], int((1-r)*L))))
#                    x = x[p:p+int(r*L), q:q+int(r*L), :]
#                    x = zoom(x, (L/int(r*L), L/int(r*L), 1), order=3)
#                    y = [round((edge[0]-p)/r), round((edge[1]-q)/r), round((edge[2]-p)/r), round((edge[3]-q)/r)]
#                for c in range(3):      # hue adjustment
#                    if c == 1:
#                        x = hue_shift(x, +random.randint(45, 75)/360.0)
#                    else:
#                        x = hue_shift(x, -random.randint(45, 75)/360.0)
#                    for d in range(2):  # gaussian noise
#                        if d == 1:
#                            gauss = np.random.normal(0,np.sqrt(random.randint(5, 10)),(L,L,3))
#                            gauss = gauss.reshape(L,L,3)
#                            x = x + gauss
##                        print(str(a)+' '+str(b)+' '+str(c)+' '+str(d))
##                        print(x.shape)
#                        idx = n*48 + a*24 + b*6 + c*2 + d
#                        X_aug[idx,:,:,:] = x
#                        Y_aug[idx,:] = y
    
    return X_aug, Y_aug


"""
Generate input and output for training
X: [N, 256, 256] - images
Y: [N, 4]        - x, y, h, w, in fraction of 256
"""
def gen_XY(labels):
    N = labels.shape[0]
    X = np.zeros((N, L, L, 3))
    Y = np.zeros((N, 4))
    for n in range(N):
        filename = labels["filename"][n]
        filename = filename[1:-1]
        im = cv2.imread(filename)
        im = im.astype('float32')
        X[n,:,:,:] = im
        x1 = labels["x_tl"][n]
        y1 = labels["y_tl"][n]
        x2 = labels["x_br"][n]
        y2 = labels["y_br"][n]
        Y[n,:] = edge2xywh([x1, y1, x2, y2])
        
    return X, Y


"""
Separate X & Y into training, devlopment & testing sets
"""
def gen_datasets(X, Y, r):
    N = Y.shape[0]
    
    N_test = round(N*r)
    N_train = N - N_test
    
    X_train = np.zeros((N_train, L,L,3))
    X_test = np.zeros((N_test, L,L,3))
    Y_train = np.zeros((N_train, 4))
    Y_test = np.zeros((N_test, 4))
    
    idx = [i for i in range(N)]
    random.seed(0)
    random.shuffle(idx)
    thresh = round(N*r)
    for i in range(N):
        if i < thresh:
            X_test[i,:,:,:] = X[idx[i],:,:,:]
            Y_test[i,:] = Y[idx[i],:]
        else:
            X_train[i-thresh,:,:,:] = X[idx[i],:,:,:]
            Y_train[i-thresh,:] = Y[idx[i],:]
    
    return X_train, X_test, Y_train, Y_test


"""
CNN model
"""
def cnn_model():
    S = 5       # kernel size in layer 1
    F = 20      # filters in conv layer 1, updated for each layer
    drop_rate = 0.4   # drop rate
    
    CL = int(np.log(L)/np.log(2) - 2)     # conv layers
    
    # Input
    x_in = Input(shape = (L,L,3), name = 'input')
    
    # layer 1: conv -> conv -> pool
    x = Conv2D(filters=F, kernel_size=(S,S), strides=(1,1), padding='same',
               kernel_initializer='he_normal')(x_in)
    x = Activation('relu')(x)
    x = Dropout(drop_rate)(x)
    x = BatchNormalization()(x)
    
#    x = Conv2D(filters=F, kernel_size=(S,S), strides=(1,1), padding='same',
#               kernel_initializer='he_normal')(x)
#    x = Activation('relu')(x)
#    x = Dropout(drop_rate)(x)
#    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

    # layers 2 - CL: conv -> conv -> pool + resnet
    for i in range(1, CL):
        if i % 2 == 1:
            F = F*1.5
        else:
            F = F*4/3
        F = int(F)
    
        # short cut
        x1 = Conv2D(filters = F, kernel_size = (1,1), strides = (1,1), padding = 'same')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x1)
        
        # conv layer
        x2 = Conv2D(filters=F, kernel_size=(S,S), padding='same',
               strides=(1,1), kernel_initializer='he_normal')(x)
        x2 = Activation('relu')(x2)
        x2 = Dropout(drop_rate)(x2)
        x2 = BatchNormalization()(x2)
        
#        x2 = Conv2D(filters=F, kernel_size=(S,S), padding='same',
#               strides=(1,1), kernel_initializer='he_normal')(x2)
#        x2 = Activation('relu')(x2)
#        x2 = Dropout(drop_rate)(x2)
#        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x2)
        x = add([x1, x2])
    
    # layer CL+1 - CL+2: dense
    x = Flatten()(x)
    U1 = F
    x = Dense(U1, activation='tanh')(x)
    x = Dropout(rate = drop_rate)(x)
    
    U2 = int(U1/3)
    x = Dense(U2, activation='tanh')(x)
    x = Dropout(rate = drop_rate)(x)
    
    # Output: softmax -> output
    x_out = Dense(4, activation='sigmoid')(x)   # need 0-1
    
    model = Model(inputs=x_in, outputs=x_out)
    adam = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-6)
    
    """
    Custom loss function
    """
    def custom_loss(y_true, y_pred):
#        x, y, w, h = y_true
#        xp, yp, wp, hp = y_pred
        loss = K.mean(K.square(y_true[...,0]-y_pred[...,0]) + K.square(y_true[...,1]-y_pred[...,1])
            + K.square(K.sqrt(y_true[...,2])-K.sqrt(y_pred[...,2]))
            + K.square(K.sqrt(y_true[...,3])-K.sqrt(y_pred[...,3])))
        return loss
    
    model.compile(optimizer=adam, loss=custom_loss)
    
    return model


"""
Train model
"""
def train_model(X_train, X_test, Y_train, Y_test):
#    batch_size = 128
    epochs = 500
    #callbacks = [EarlyStopping(monitor='val_loss', patience=7)]
    callbacks = [ModelCheckpoint(filepath = 'weights-best.hdf5', monitor='val_loss', save_best_only=True)]
    
    # Load model
    model = cnn_model()
    
    # Train model
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
              epochs = epochs, shuffle = True, callbacks = callbacks)

    return model


"""
Intersection over union
Return the mean & std of IoU in a group
"""
def iou(Y, Y_pred):
    N = Y.shape[0]
    IoU = np.zeros((N,1))
    for n in range(N):
        xywh = Y[n,:]
        xywh_pred = Y_pred[n,:]
        edge = xywh2edge(xywh)
        edge_pred = xywh2edge(xywh_pred)
        A = (edge[2]-edge[0]+1) * (edge[3]-edge[1]+1)
        A_pred = (edge_pred[2]-edge_pred[0]+1) * (edge_pred[3]-edge_pred[1]+1)
        
        x1 = max(edge[0], edge_pred[0])
        y1 = max(edge[1], edge_pred[1])
        x2 = min(edge[2], edge_pred[2])
        y2 = min(edge[3], edge_pred[3])
        A_inter = max(0, x2-x1+1) * max(0, y2-y1+1)
        iou = A_inter / float(A + A_pred - A_inter)
        IoU[n] = iou
    
    mean_iou = np.mean(IoU)
    std_iou = np.std(IoU)
    return mean_iou, std_iou


"""
Evaluate model performance
"""
def eval_model(model, X_train, X_dev, X_test, Y_train, Y_dev, Y_test):
    Y_train_pred = model.predict(X_train)
    Y_dev_pred = model.predict(X_dev)
    Y_test_pred = model.predict(X_test)
    mean_train, std_train = iou(Y_train, Y_train_pred)
    mean_dev, std_dev = iou(Y_dev, Y_dev_pred)
    mean_test, std_test = iou(Y_test, Y_test_pred)
    # Training set performance
#    N = Y_train.shape[0]
#    for n in range(N):
#        im = X_train[n,:,:,:]
#        img = Image.fromarray(np.uint8(im))
#        draw = Draw(img)
#        coord = Y_train[n][:]
#        box_truth = xywh2edge(coord)
#        coord = Y_train_pred[n][:]
#        box_pred = xywh2edge(coord)
#        draw.rectangle(box_truth, outline='red')
#        draw.rectangle(box_pred, outline='green')
#        break
    return mean_train, std_train, mean_dev, std_dev, mean_test, std_test
    


"""
Driver
"""
if __name__ == "__main__":
    datadir = "../Cropped/"
    label_file = "../labels.csv"
    global_params()
    labels = pd.read_csv(label_file)
    X, Y = gen_XY(labels)
    X_train, X_test, Y_train, Y_test = gen_datasets(X, Y, 0.1)
    print("augmenting data")
    t = time.time()
    X_aug, Y_aug = augment_data(X_train, Y_train)
    X_train, X_dev, Y_train, Y_dev = gen_datasets(X_aug, Y_aug, 0.1111)
    t = time.time() - t
    print("done generating data, t = " + str(round(t)))
    
    model = train_model(X_train, X_dev, Y_train, Y_dev)
    t = time.time() - t
    print("done training, t = " + str(round(t)))
    mean_train, std_train, mean_dev, std_dev, mean_test, std_test = eval_model(model, X_train, X_dev, X_test, Y_train, Y_dev, Y_test)
    
    
