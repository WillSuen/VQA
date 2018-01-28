import numpy as np
import os
import shutil
import glob
import os
import re
import lmdb
import pickle
import h5py
import mxnet as mx
import sqlite3
import io
import cv2

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def get_image(img_path, show=False):
    # download and show the image
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    if img is None:
         return None
    if show:
         plt.imshow(img)
         plt.axis('off')
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (448, 448))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img

# download resent-152
path='http://data.mxnet.io/models/imagenet-11k/'
[mx.test_utils.download(path+'resnet-152/resnet-152-symbol.json'),
 mx.test_utils.download(path+'resnet-152/resnet-152-0000.params'),
 mx.test_utils.download(path+'synset.txt')]

sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-152', 0)
mod = mx.mod.Module(symbol=sym, context=mx.gpu(0), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1, 3, 448, 448))], 
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)
with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]
    

interals = sym.get_internals()
_, out_shapes, _ = interals.infer_shape(data=(1, 3, 448, 448))
shape_dict = dict(zip(interals.list_outputs(), out_shapes))
    
img_root = '/home/wsun12/src/datasets/vqa2.0/train2014/'
imgs_train = glob.glob('/home/wsun12/src/datasets/vqa2.0/train2014/*.jpg')
imgs_test = glob.glob('/home/wsun12/src/datasets/vqa2.0/val2014/*.jpg')
N_train = len(imgs_train)
N_test = len(imgs_test)

print('train image size', len(imgs_train))
print('validation image size', len(imgs_test))

all_layers = sym.get_internals()
fe_sym = all_layers['_plus49_output']
fe_mod = mx.mod.Module(symbol=fe_sym, context=mx.gpu(0), label_names=None)
fe_mod.bind(for_training=False, data_shapes=[('data', (1, 3, 448, 448))])
fe_mod.set_params(arg_params, aux_params)


## connect sqlite3
# Converts np.array to TEXT when inserting
print("Now connect to db...")
sqlite3.register_adapter(np.ndarray, adapt_array)
# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)
con = sqlite3.connect("img_features.db", detect_types=sqlite3.PARSE_DECLTYPES)
cur = con.cursor()
cur.execute('drop table if exists features')
cur.execute("create table features (img_path string primary key, arr array)")

print("Start to extract features from images...")
for i, img in enumerate(imgs_train):
    label = img[33:]
    img_array = get_image(img)
    fe_mod.forward(mx.io.DataBatch([mx.nd.array(img_array)]))
    feature = fe_mod.get_outputs()[0].asnumpy()
    cur.execute("insert into features values (?, ?)", (label, feature))
    if (i+1) % 1000 == 0:
        con.commit()
        cur.execute("select count(*) from features")
        print(str(i+1) + ' of '+ str(N_train) + ' images done!', "In database:", cur.fetchone()[0])

        
for i, img in enumerate(imgs_test):
    label = img[33:]
    img_array = get_image(img)
    fe_mod.forward(mx.io.DataBatch([mx.nd.array(img_array)]))
    feature = fe_mod.get_outputs()[0].asnumpy()
    cur.execute("insert into features values (?, ?)", (label, feature))  
    if (i+1) % 1000 == 0:
        con.commit()
        cur.execute("select count(*) from features")
        print(str(i+1) + ' of '+ str(N_test) + ' images done!', "In database:", cur.fetchone()[0])

con.commit()
cur.execute("select count(*) from features")
print "In database:", cur.fetchone()[0]
con.close()
