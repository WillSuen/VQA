import mxnet as mx
import numpy as np
import os

from collections import namedtuple
# use bayesian GRU
GRUState = namedtuple("GRUState", ["h"])
GRUParam = namedtuple("GRUParam", ["gates_i2h_weight", "gates_i2h_bias",
                                   "gates_h2h_weight", 
                                   "trans_i2h_weight", "trans_i2h_bias",
                                   "trans_h2h_weight"])
GRUDropoutParam = namedtuple("GRUDropoutParam", ["gates_i2h", "gates_h2h", 
                                   "trans_i2h", "trans_h2h"])


LSTMState = namedtuple("LSTMState", ["h", "c"])
LSTMParam = namedtuple("LSTMParam", ["gates_i2h_weight", "gates_i2h_bias",
                                     "gates_h2h_weight", "gates_h2h_bias"])

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx,
        prefix, mask=None, dp_param=None):
    if dp_param > 0:
        indata = mx.sym.Dropout(indata, p=dp_param)
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.gates_i2h_weight,
                                bias=param.gates_i2h_bias,
                                num_hidden=num_hidden*4,
                                name='%s_t%d_l%d_gates_i2h'%(prefix, seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.gates_h2h_weight,
                                bias=param.gates_h2h_bias,
                                num_hidden=num_hidden*4,
                                name='%s_t%d_l%d_gates_h2h'%(prefix, seqidx, layeridx))
    gates = i2h + h2h
    gates = mx.sym.split(gates, num_outputs=4,
                         name='%s_t%d_l%d_slide'%(prefix, seqidx, layeridx))
    forget_gate = mx.sym.Activation(gates[0], act_type='sigmoid')
    input_gate = mx.sym.Activation(gates[1], act_type='sigmoid')
    output_gate = mx.sym.Activation(gates[2], act_type='sigmoid')
    
    cell_state = mx.sym.Activation(gates[3], act_type='tanh')
    next_c = forget_gate * prev_state.c + input_gate * cell_state
    next_h = output_state * next_c
    if dp_param > 0:
        next_h = mx.sym.Dropout(next_h, p=dp_param)
    if mask is not None:
        next_h = prev_state.h + mx.sym.broadcast_mul(mask, next_h - prev_state.h)
    return LSTMState(h=next_h, c=next_c)


def sru(num_hidden, indata, prev_state, param, seqidx, layeridx,
        prefix, mask=None, dp_param=None):
    ## https://arxiv.org/pdf/1709.02755.pdf
    ## Simple Recurrent Unit (SRU), an rnn architecture fast as CNNs
    if dp_param > 0:
        indata = mx.sym.Dropout(indata, p=dp_param)
    i2i = mx.sym.FullyConnected(data=indata,
                                weight=param.gates_i2i_weight,
                                no_bias=True,
                                num_hidden=num_hidden,
                                name='%s_t%d_l%d_gates_i2i'%(prefix, seqidx, layeridx))
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.gates_i2h_weight,
                                bias=param.gates_i2h_bias,
                                num_hidden=num_hidden*2,
                                name='%s_t%d_l%d_gates_i2h'%(prefix, seqidx, layeridx))
    gates = i2h
    gates = mx.sym.split(gates, num_outputs=4,
                         name='%s_t%d_l%d_slide'%(prefix, seqidx, layeridx))
    forget_gate = mx.sym.Activation(gates[0], act_type='sigmoid')
    reset_gate = mx.sym.Activation(gates[1], act_type='sigmoid')
    
    next_c = forget_gate * prev_state.c + (1 - forget_gate) * i2i
    next_h = reset_gate * mx.sym.Activation(next_c, act_type='tanh') + (1 - reset_gate)*indata
    if dp_param > 0:
        next_h = mx.sym.Dropout(next_h, p=dp_param)
    if mask is not None:
        next_h = prev_state.h + mx.sym.broadcast_mul(mask, next_h - prev_state.h)
    return LSTMState(h=next_h, c=next_c)


def LSTM_unroll(batch_size, input_seq, in_dim, seq_len, num_hidden, prefix, 
               dropout=0, mask=None, n_gpus=1):
    """
    Data:
    prefix+'l0_init_h' and 'l0_init_c': set to all 0
    mask: used for variable length sequences
    need_middle: whether we need the intermediate h. For sentence embedding, set to False.
                 for input module, set to True
    """
    dp_param = dropout
    layer_num = 0
    lstm_param = LSTMParam(gates_i2h_weight=mx.sym.Variable('%s_l%d_i2h_gates_weight' % (prefix, layer_num)),
                           gates_i2h_bias=mx.sym.Variable("%s_l%d_i2h_gates_bias" % (prefix, layer_num)),
                           gates_h2h_weight=mx.sym.Variable("%s_l%d_h2h_gates_weight" % (prefix, layer_num))
                          )
    state = LSTMState(h=mx.sym.Variable("%s_l%d_init_h" % (prefix, layer_num)),
                      c=mx.sym.Variable("%s_l%d_init_c" % (prefix, layer_num)))
    wordvec = mx.sym.split(data=input_seq, num_outputs=seq_len, 
                           squeeze_axis=True, name=prefix+'_slice_word')
    masks = mx.sym.split(data=mask, num_outputs=seq_len, 
                         squeeze_axis=False, name=prefix+'_slice_mask')
    
    # get first output
    hidden = wordvec[0]
    mask_t = masks[0]  
    state = lstm(num_hidden, indata=hidden, mask=mask_t,
                 prev_state=state, param=lstm_param,
                 seqidx=0, layeridx=layer_num, 
                 dp_param=dp_param, prefix=prefix)
    
    ret = mx.sym.reshape(data=state.h, shape=(batch_size, 1, num_hidden))
    # get remaining outputs
    for seqidx in range(1, seq_len):
        hidden = wordvec[seqidx]
        mask_t = masks[seqidx]
        
        state = lstm(num_hidden, indata=hidden, mask=mask_t,
                     prev_state=state, param=lstm_param,
                     seqidx=seqidx, layeridx=layer_num, 
                     dp_param=dp_param, prefix=prefix)
        ret = mx.sym.concat(ret, mx.sym.reshape(data=state.h, shape=(batch_size, 1, num_hidden)), dim=1)
        
    return ret


def gru(num_hidden, indata, prev_state, param, seqidx, layeridx, 
        prefix, mask=None, dp_param=None):
    # mask=1, update h; otherwise, keep h
    if dp_param is not None:
        indata_gates = mx.sym.broadcast_mul(indata, dp_param.gates_i2h)
        prevh_gates = mx.sym.broadcast_mul(prev_state.h, dp_param.gates_h2h)
        indata_trans = mx.sym.broadcast_mul(indata, dp_param.trans_i2h)
    else:
        indata_gates = indata
        prevh_gates = prev_state.h
        indata_trans = indata


    i2h = mx.sym.FullyConnected(data=indata_gates, 
                                weight=param.gates_i2h_weight,
                                bias=param.gates_i2h_bias,
                                num_hidden=num_hidden*2,
                                name='%s_t%d_l%d_gates_i2h'%(prefix, seqidx, layeridx))
    # use encoder_U
    h2h = mx.sym.FullyConnected(data=prevh_gates,
                                weight=param.gates_h2h_weight,
                                no_bias=True,
                                num_hidden=num_hidden*2,
                                name='%s_t%d_l%d_gates_h2h'%(prefix, seqidx, layeridx))
    gates = i2h+h2h

    gates_act = mx.sym.Activation(gates, act_type='sigmoid')
    slice_gates = mx.sym.split(gates_act, num_outputs=2,
                               name='%s_t%d_l%d_slide'%(prefix, seqidx, layeridx))
    
    update_gate = slice_gates[0]
    reset_gate = slice_gates[1]

    htrans_i2h = mx.sym.FullyConnected(data=indata_trans, 
                                       weight=param.trans_i2h_weight,
                                       bias=param.trans_i2h_bias,
                                       num_hidden=num_hidden,
                                       name='%s_t%d_l%d_trans_i2h'%(prefix, seqidx, layeridx))

    h_after_reset = prev_state.h * reset_gate
    # use encode_Ux
    if dp_param is not None:
        h_after_reset = mx.sym.broadcast_mul(h_after_reset, dp_param.trans_h2h)

    htrans_h2h = mx.sym.FullyConnected(data=h_after_reset, 
                                       weight=param.trans_h2h_weight,
                                       no_bias=True,
                                       num_hidden=num_hidden,
                                       name='%s_t%d_l%d_trans_h2h'%(prefix, seqidx, layeridx))
    h_trans = htrans_i2h + htrans_h2h
    h_trans_active = mx.sym.Activation(h_trans, act_type='tanh')
    next_h = prev_state.h + update_gate * (h_trans_active - prev_state.h)
    if mask is not None:
        next_h = prev_state.h + mx.sym.broadcast_mul(mask, next_h - prev_state.h)
    return GRUState(h=next_h)


def GRU_unroll(batch_size, input_seq, in_dim, seq_len, num_hidden, prefix, 
               dropout=0, mask=None, n_gpus=1):
    """
    Data:
    prefix+'l0_init_h': set to all 0
    mask: used for variable length sequences
    need_middle: whether we need the intermediate h. For sentence embedding, set to False.
                    for input module, set to True
    """
    if dropout>0:
        x0 = batch_size if n_gpus==1 else 1 # unsolved problem here...
        dp_param = GRUDropoutParam(gates_i2h=bayesian_dp_sym(dropout, (x0, in_dim)),
                                   gates_h2h=bayesian_dp_sym(dropout, (x0, num_hidden)),
                                   trans_i2h=bayesian_dp_sym(dropout, (x0, in_dim)),
                                   trans_h2h=bayesian_dp_sym(dropout, (x0, num_hidden)))
    else:
        dp_param=None

    layer_num = 0
    gru_param = GRUParam(gates_i2h_weight=mx.sym.Variable('%s_l%d_i2h_gates_weight'%(prefix, layer_num)),
                         gates_i2h_bias=mx.sym.Variable("%s_l%d_i2h_gates_bias" % (prefix, layer_num)),
                         gates_h2h_weight=mx.sym.Variable("%s_l%d_h2h_gates_weight" % (prefix, layer_num)),
                         trans_i2h_weight=mx.sym.Variable("%s_l%d_i2h_trans_weight" % (prefix, layer_num)),
                         trans_i2h_bias=mx.sym.Variable("%s_l%d_i2h_trans_bias" % (prefix, layer_num)),
                         trans_h2h_weight=mx.sym.Variable("%s_l%d_h2h_trans_weight" % (prefix, layer_num)))
    state = GRUState(h=mx.sym.Variable("%s_l%d_init_h" % (prefix, layer_num)))
    wordvec = mx.sym.split(data=input_seq, num_outputs=seq_len, 
                                  squeeze_axis=True, name=prefix+'_slice_word')
    masks = mx.sym.split(data=mask, num_outputs=seq_len, 
                                squeeze_axis=False, name=prefix+'_slice_mask')
    
    # get first output
    hidden = wordvec[0]
    mask_t = masks[0]  
    state = gru(num_hidden, indata=hidden, mask=mask_t,
                prev_state=state, param=gru_param,
                seqidx=0, layeridx=layer_num, 
                dp_param=dp_param, prefix=prefix)
    
    ret = mx.sym.reshape(data=state.h, shape=(batch_size, 1, 512))
    # get remaining outputs
    for seqidx in range(1, seq_len):
        hidden = wordvec[seqidx]
        mask_t = masks[seqidx]
        
        state = gru(num_hidden, indata=hidden, mask=mask_t,
                    prev_state=state, param=gru_param,
                    seqidx=seqidx, layeridx=layer_num, 
                    dp_param=dp_param, prefix=prefix)
        ret = mx.sym.concat(ret, mx.sym.reshape(data=state.h, shape=(batch_size, 1, 512)), dim=1)
        
    return ret


def bayesian_dp_sym(p, shape):
    # make a dropout mask for bayesian dropout
    # calculate the lower bound corresponding to bayesian dp
    assert(p<=0.5)
    uni_min = (0.5 - p)/(1.0 - p)
    rand_num = mx.sym.uniform(low=uni_min, high=1, shape=shape)
    mask = mx.sym.round(rand_num) / (1.0 - p)
    return mask


def parallel_attention(ques_feat, img_feat, mask, batch_size):
    ## parallel_attention
    img_seq_size = 196
    ques_seq_size = 25
    hidden_size = 512

    img_corr = mx.sym.FullyConnected(data=img_feat, num_hidden=hidden_size, flatten=False)
    
    # weight_matrix shape will be batch_size X ques_seq_size X img_seq_size
    weight_matrix = mx.sym.batch_dot(lhs=ques_feat, rhs=img_corr, transpose_b=True)
    weight_matrix = mx.sym.Activation(data=weight_matrix, act_type='tanh')

    ques_embed = mx.sym.FullyConnected(data=ques_feat, num_hidden=hidden_size, flatten=False)
    img_embed = mx.sym.FullyConnected(data=img_feat, num_hidden=hidden_size, flatten=False)

    # atten for question feature
    transform_img = mx.sym.batch_dot(lhs=weight_matrix, rhs=img_embed)
    ques_atten_sum = mx.sym.Activation(data=transform_img+ques_embed, act_type='tanh')
    ques_atten_sum = mx.sym.Dropout(data=ques_atten_sum, p=0.5)
    ques_atten = mx.sym.FullyConnected(data=ques_atten_sum, num_hidden=1, flatten=False)

    ## mask softmax for question
    ques_atten = mx.sym.broadcast_mul(mx.sym.reshape(ques_atten, shape=(batch_size, -1)), mask)
    ques_atten = ques_atten + (1-mask)*(-9999)     # softmax output of -9999 would be almost 0
    ques_atten = mx.sym.softmax(data=ques_atten)

    # atten for image feature
    transform_ques = mx.sym.batch_dot(lhs=weight_matrix, rhs=ques_embed, transpose_a=True)
    img_atten_sum = mx.sym.Activation(data=transform_ques+img_embed, act_type='tanh')
    img_atten_sum = mx.sym.Dropout(data=img_atten_sum, p=0.5)

    img_atten_embedding = mx.sym.FullyConnected(data=img_atten_sum, num_hidden=1, flatten=False)
    img_atten = mx.sym.softmax(data=mx.sym.reshape(data=img_atten_embedding, shape=(batch_size, -1)))

    # reshape to 3d array
    ques_atten = mx.sym.reshape(data=ques_atten, shape=(batch_size, 1, -1))
    img_atten = mx.sym.reshape(data=img_atten, shape=(batch_size, 1, -1))

    ques_atten_feat = mx.sym.batch_dot(lhs=ques_atten, rhs=ques_feat)
    ques_atten_feat = mx.sym.reshape(data=ques_atten_feat, shape=(batch_size, hidden_size))

    img_atten_feat = mx.sym.batch_dot(lhs=img_atten, rhs=img_feat)
    img_atten_feat = mx.sym.reshape(data=img_atten_feat, shape=(batch_size, hidden_size))
    
    return ques_atten_feat, img_atten_feat


def get_symbol(cfg):
    # constants
    batch_size = cfg.batch_size
    rnn_type = cfg.network.rnn_type
    hidden_size = cfg.network.hidden_size
    vocab_size = cfg.network.vocab_size
   
    wordembed_dim = hidden_size    # output dimension for word embedding
    
    # image feature
    ifeature = mx.sym.Variable('img_feature')    # input img_feature: batch_size X 196 X 2048
    hidden1 = mx.sym.FullyConnected(data=ifeature, num_hidden=hidden_size, name='FC1', flatten=False)
    act1 = mx.sym.Activation(data=hidden1, act_type='tanh', name='Act1')
    drop1 = mx.sym.Dropout(data=act1, p=0.5, name='Drop1')    # output from this cnn: batch_size X 196 X hidden_size  
    
    # question feature
    sent_seq = mx.sym.Variable('sent_seq')
    mask = mx.sym.Variable('mask')
    embed_weight = mx.sym.Variable('embed_weight')

    # oputput dimension for question: batch_size X ques_seq_length X wordembed_dim
    embeded_seq = mx.sym.Embedding(data=sent_seq, input_dim=vocab_size, weight=embed_weight, 
                                   output_dim=wordembed_dim, name='sent_embedding')
    
    w_ques, w_img = parallel_attention(embeded_seq, drop1, mask, batch_size)
    
    
    ## phrase level
    img_seq_size = 196
    ques_seq_size = 25
    
    unigram = mx.sym.Convolution(data=embeded_seq, kernel=(1), pad=(0,), num_filter=ques_seq_size)
    bigram = mx.sym.Convolution(data=embeded_seq, kernel=(2), pad=(1,), num_filter=ques_seq_size)
    trigram = mx.sym.Convolution(data=embeded_seq, kernel=(3), pad=(1,), num_filter=ques_seq_size)
    bigram = mx.sym.slice(data=bigram, begin=(0, 0, 1), end=(batch_size, ques_seq_size, hidden_size+1))

    unigram = mx.sym.reshape(data=unigram, shape=(batch_size, ques_seq_size, hidden_size, -1))
    bigram = mx.sym.reshape(data=bigram, shape=(batch_size, ques_seq_size, hidden_size, -1))
    trigram = mx.sym.reshape(data=trigram, shape=(batch_size, ques_seq_size, hidden_size, -1))

    feat = mx.sym.concat(unigram, bigram, trigram, dim=3)
    max_feat = mx.sym.Activation(data=mx.sym.max(data=feat, axis=3), act_type='tanh')
    max_feat = mx.sym.Dropout(data=max_feat, p=0.5)
    
    p_ques, p_img = parallel_attention(max_feat, drop1, mask, batch_size)
    
    
    ## sentence level
    if rnn_type == 'gru':
        sent_vec = GRU_unroll(batch_size, embeded_seq, mask=mask, 
                              in_dim=wordembed_dim, seq_len=ques_seq_size, 
                              num_hidden=hidden_size, dropout=0.5,
                              prefix='sent', n_gpus=1)
    else:
        sent_vec = LSTM_unroll(batch_size, embeded_seq, mask=mask, 
                              in_dim=wordembed_dim, seq_len=ques_seq_size, 
                              num_hidden=hidden_size, dropout=0.5,
                              prefix='sent', n_gpus=1)
    q_ques, q_img = parallel_attention(sent_vec, drop1, mask, batch_size)
    
    
    # final feature extraction
    feat1 = mx.sym.Dropout(data=w_ques+w_img, p=0.5)
    hidden1 = mx.sym.FullyConnected(data=feat1, num_hidden=hidden_size)
    hidden1 = mx.sym.Activation(data=hidden1, act_type='tanh')
    feat2 = mx.sym.Dropout(data=mx.sym.concat(p_ques+p_img, hidden1, dim=1), p=0.5)
    hidden2 = mx.sym.FullyConnected(data=feat2, num_hidden=hidden_size)
    hidden2 = mx.sym.Activation(data=hidden2, act_type='tanh')
    feat3 = mx.sym.Dropout(data=mx.sym.Concat(q_img+q_ques, hidden2, dim=1), p=0.5)
    hidden3 = mx.sym.FullyConnected(data=feat3, num_hidden=hidden_size)
    hidden3 = mx.sym.Activation(data=hidden3, act_type='tanh')
    outfeat = mx.sym.FullyConnected(data=mx.sym.Dropout(data=hidden3, p=0.5), num_hidden=1000)
    
    label = mx.sym.Variable('ans_label')
    out_fc = mx.sym.FullyConnected(data=outfeat, num_hidden=2000, name='out_fc')
    out_score = mx.sym.SoftmaxOutput(data=out_fc, label=label, name='out_sm')
    
    return out_score
    