import mxnet as mx
import numpy as np
import numpy.matlib
import numpy.random
import os
import json, pickle
import random
import logging
import lmdb
import time
import re
import sqlite3
import io


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

class SimpleBatch(object):
  def __init__(self, data_names, data, label_names, label, 
               bucket_key=None, qid=None, ans_all=None, splits=None):
    self.data = data
    self.label = label
    self.data_names = data_names
    self.label_names = label_names
    self.bucket_key = bucket_key

    self.pad = 0
    self.splits=splits # a list of splits
    self.index = None
    self.qid = qid # should be a list of qid's
    self.ans_all = ans_all

    self.provide_data = [(n, x.shape) for n,x in zip(self.data_names, self.data)]
    self.provide_label = [(n, x.shape) for n,x in zip(self.label_names, self.label)]

    
class VQAIter(mx.io.DataIter):
    def __init__(self, cfg, is_train=True, net=None):
        """
        Data loader for the VQA dataset.abs

        qa_path: path to the question-answer file
        lmdb_path: the LMDB storing the extracted features
        net: symbol of the network, to print its size
        is_train: use answer sampling if set to True
        """
        super(VQAIter, self).__init__()
        random.seed(cfg.dataset.seed)
        qa_paths = cfg.dataset.qa_path.split(',')
        logging.info("QA data paths:{}".format(qa_paths))
        self.rnn_type = cfg.network.rnn_type
        self.batch_size = cfg.batch_size
        self.is_train = is_train
        sent_gru_hsize = cfg.dataset.sent_gru_hsize
        max_seq_len = cfg.dataset.max_seq_length
        w = cfg.dataset.w
        h = cfg.dataset.h
        
        # read image features from database
        sqlite3.register_adapter(np.ndarray, adapt_array)
        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("array", convert_array)
        self.con = sqlite3.connect(cfg.dataset.img_feature_path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.cur = self.con.cursor()

        # whether to use snake-shaped image data
        self.provide_data = [('img_feature', (self.batch_size, w*h, 2048)),
                             ('sent_seq', (self.batch_size, max_seq_len)),
                             ('mask', (self.batch_size, max_seq_len)),
                             ('sent_l0_init_h', (self.batch_size, 512))]
        if self.rnn_type == 'lstm':
            self.provide_data.append(('sent_l0_init_c', (self.batch_size, 512)))
        
        self.provide_label = [('ans_label', (self.batch_size,))]

        self.data_names = [t[0] for t in self.provide_data]
        self.label_names = [t[0] for t in self.provide_label]
        self.data_buffer = [np.zeros(t[1], dtype=np.float32) for t in self.provide_data]
        self.label_buffer = [np.zeros(t[1], dtype=np.float32) for t in self.provide_label]
        
        self.qa_list = []
        for path in qa_paths:
            self.qa_list += pickle.load(open(path))
        
        self.qa_list = self.qa_list[:400000]
        
        # print self.provide_data
        if net is not None:
            shape_list = net.infer_shape(**dict(self.provide_data+self.provide_label))
            arg_names = net.list_arguments()
            n_params = 0
            logging.info("Number of parameters:")
            for n, shape in enumerate(shape_list[0]):
                if arg_names[n] not in self.data_names and arg_names[n] not in self.label_names:
                    logging.info("%s: %d, i.e., %.2f M params", arg_names[n], np.prod(shape), np.prod(shape)/1e6)
                    n_params += np.prod(shape)
            logging.info("Total number of parameters:%d, i.e., %.2f M params", n_params, n_params/1e6)

        self.last_batch_size=None # signaling the changed batch size
        self.n_total = len(self.qa_list)
        self.reset()

    def reset(self):
        if self.is_train:
            logging.info("Shuffling data...")
            random.shuffle(self.qa_list)

    def __iter__(self):
        candidate_ans = np.zeros((self.batch_size, 10), dtype=np.int32)
        
        for curr_idx in range(0, self.n_total-self.batch_size+1, self.batch_size):
            qid_list=[]
            for bidx in range(self.batch_size):
                bdata = self.qa_list[bidx+curr_idx]
                img_label = bdata['img_path']
                self.cur.execute("select arr from features where img_path = '%s'" % img_label)
                img_feature = self.cur.fetchone()[0].reshape(2048, -1)
                
                self.data_buffer[0][bidx,:,:] = np.swapaxes(img_feature, 0, 1)
                self.data_buffer[1][bidx, :] = bdata['ques']
                self.data_buffer[2][bidx, :] = bdata['qmask']
                
                qid_list.append(bdata['qid'])
                if self.is_train:
                    self.label_buffer[0][bidx] = np.random.choice(bdata['ans_cans'], p=bdata['ans_p'])
                #if not self.is_train and len(bdata['ans_all'])>0:
                    # for VQA validation only
                    #candidate_ans[bidx] = bdata['ans_all']

            yield SimpleBatch(self.data_names, [mx.nd.array(arr) for arr in self.data_buffer],
                              self.label_names, [mx.nd.array(arr) for arr in self.label_buffer], 
                              qid=qid_list)#, ans_all=candidate_ans)

        # check if need to add an incomplete batch at validation
        if not self.is_train and curr_idx < self.n_total-self.batch_size:
            curr_idx += self.batch_size
            self.last_batch_size = self.n_total - curr_idx
            print("last_batch_size {}".format(self.last_batch_size))
            candidate_ans = np.zeros((self.last_batch_size, 10), dtype=np.int32)
            # change the shape of buffer files
            data_buffer=[np.zeros([self.last_batch_size]+list(shape[1:])) for name, shape in self.provide_data]
            label_buffer=[np.zeros([self.last_batch_size]+list(shape[1:])) for name, shape in self.provide_label]
            qid_list=[]
            for bidx in range(self.last_batch_size):
                bdata = self.qa_list[bidx+curr_idx]
                
                label = bdata['img_path']
                self.cur.execute("select arr from features where img_path = '%s'" % label)
                feature = self.cur.fetchone()[0].reshape(2048, -1)
                
                data_buffer[0][bidx,:,:] = np.swapaxes(feature, 0, 1)
                data_buffer[1][bidx, :] = bdata['ques']
                data_buffer[2][bidx, :] = bdata['qmask']
                
                qid_list.append(bdata['qid'])
                #if len(bdata['ans_all'])>0:
                    #label_buffer[0][bidx] = np.random.choice(bdata['ans_cans'], p=bdata['ans_p'])
                    #candidate_ans[bidx] = bdata['ans_all']

            yield SimpleBatch(self.data_names, [mx.nd.array(arr) for arr in data_buffer],
                              self.label_names, [mx.nd.array(arr) for arr in label_buffer], 
                              qid=qid_list)#, ans_all=candidate_ans)
