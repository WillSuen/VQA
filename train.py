import argparse,logging,os
import mxnet as mx
import symbol.hieSymbols as hievqa
import symbol.hieLoaders as hieloader
from cfgs.config import cfg, read_cfg
import pprint
import numpy as np
import pickle


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    read_cfg(args.cfg)
    if args.gpus:
        cfg.gpus = args.gpus
    if args.model_path:
        cfg.model_path = args.model_path
    pprint.pprint(cfg)
    
    # get symbol
    symbol = hievqa.get_symbol(cfg)
    kv = mx.kvstore.create(cfg.kv_store)
    devs = mx.cpu() if cfg.gpus is None else [mx.gpu(int(i)) for i in cfg.gpus.split(',')]
    begin_epoch = cfg.model_load_epoch if cfg.model_load_epoch else 0
    if not os.path.exists(cfg.model_path):
        os.mkdir(cfg.model_path)
    model_prefix = cfg.model_path + "hierarchical_VQA"
    checkpoint = mx.callback.do_checkpoint(model_prefix)
    
    # data iter
    train_iter = hieloader.VQAIter(cfg)

    if cfg.train.lr_factor_epoch>0:
        step = cfg.train.lr_factor_epoch*(train_iter.n_total // cfg.batch_size)
    else:
        step=1
    opt_args = {}
    opt_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(step=step, factor=cfg.train.lr_factor)
    optimizer = mx.optimizer.Adam(learning_rate=cfg.train.lr, beta1=cfg.train.beta1, beta2=cfg.train.beta2, 
                                  wd=cfg.train.wd, **opt_args)

    model = mx.mod.Module(context=devs, symbol=symbol, data_names=train_iter.data_names,
                          label_names=train_iter.label_names)
    
    if cfg.retrain:
        _, arg_params, __ = mx.model.load_checkpoint(model_prefix, cfg.model_load_epoch)
    else:
        # containing only the skip thought weights
        arg_params = pickle.load(open(cfg.train.skip_thought_dict))

    embed_param = {}
    embed_param['embed_weight'] = arg_params['embed_weight']
    initializer = mx.initializer.Load(embed_param, 
                                      default_init=mx.initializer.Uniform(cfg.train.uni_mag), 
                                      verbose=True)

    def top1_accuracy(labels, preds):
        pred_labels = np.argmax(preds, axis=1)
        n_correct = np.where(labels==pred_labels)[0].size
        return n_correct/np.float32(labels.size)

    metrics = [mx.metric.CrossEntropy(), mx.metric.CustomMetric(top1_accuracy, allow_extra_outputs=True)]
    epoch_end_callback = [mx.callback.do_checkpoint(model_prefix, 1)]#, test_callback]
    batch_end_callback = [mx.callback.Speedometer(cfg.batch_size, cfg.frequent)]

    print('=================================================================================')
    print('Start training...')
    model.fit(train_data=train_iter,
              eval_metric=mx.metric.CompositeEvalMetric(metrics=metrics),
              epoch_end_callback=epoch_end_callback,
              batch_end_callback=batch_end_callback,
              optimizer=optimizer,
              # initializer=initializer,
              begin_epoch=cfg.model_load_epoch,
              num_epoch=cfg.num_epoch)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Hierarchical Visual Attention on the VQA dataset')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--model_path', help='the loc to save model checkpoints', default='', type=str)
    parser.add_argument('--skip-thought-dict', type=str, default='/home/wsun12/src/VQA/vqa-sva/rundata/skip-argdict.pkl',
                        help='initialize the GRU for skip-thought vector')

    args = parser.parse_args()
    logging.info(args)
    main()
