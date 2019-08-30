import json
import os
import os.path
import glob

import numpy as np
import tensorflow as tf

from utils.load_signals import LoadSignals
from utils.prep_data import train_val_loo_split, train_val_test_split, train_val_split
from utils.log import log
from models.cnn import ConvNN
from models.cnn_gan import CNNGAN, CNNGAN_infer
from dcgan.model import DCGAN

def makedirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass


def main(dataset='Kaggle2014Pred', build_type='cv', sph=5):
    print ('Main')
    with open('SETTINGS_%s.json' %dataset) as f:
        settings = json.load(f)

    makedirs(str(settings['cachedir']))
    makedirs(str(settings['resultdir']))
    makedirs(str(settings['ganckptdir']))
    makedirs(str(settings['cnnckptdir']))
   
    if settings['dataset']=='FB':
        targets = [
            '1',
            # '3',
            # '4',
            # '5',
            # '6',
            # '13',
            # '14',
            # '15',
            # '16',
            # '17',
            # '18',
            # '19',
            # '20',
            # '21'
        ]
    elif settings['dataset']=='CHBMIT': 
        # exclude patients have too frequent seizures
        targets = [
            '1',
            # '2',
            # '3',          
            # '5',           
            # '9',
            # '10',           
            # '13',
            # '14',            
            # '18',
            # '19',
            # '20',
            # '21',            
            # '23'
        ]
    elif settings['dataset']=='EpilepsiaSurf':
        targets = [
            '1',
            # '2',
            # '3',
            # '4',
            # '5',

            # '6',
            # '7',
            # '8',
            # '9',
            # '10',
            # '11',
            # '12',


            # '13',
            # '14',
            # '15',
            # '16',
            # '17',
            # '18',
            #'19',

            # '20',
            # '21',
            # '22',
            # '23',
            # '24',
            # '25',
            # '26',
            # '27',
            # '28',
            # '29',
            # '30'
        ]
    summary = {}
    lines = ['clip,seizure']
    for target in targets:

        if build_type=='save_STFT':
            dir = str(settings['stftdir']) + '/STFT_%s_%d' %(dataset,sph)
            makedirs(dir)
            LoadSignals(target, type='ictal', settings=settings, sph=sph).apply(
                save_STFT=True, over_spl=True, # set over_spl=True to generate more samples for GAN training
                dir=dir
            )
            LoadSignals(target, type='interictal', settings=settings, sph=sph).apply(
                save_STFT=True, over_spl=True,
                dir=dir
            )
        elif build_type=='dcgan':

            checkpoint = settings['ganckptdir'] + "/%s" % target # need to change dcgan/model.py pattern as well
            makedirs(checkpoint)
            stft_dirs = []
            stft_dirs.append(settings['stftdir'] + '/STFT_%s_%d' %(dataset,sph))
            #stft_dirs.append(settings['stftdir2'] + '/STFT_%s_%d' %(dataset,sph)) # in case not enough space storing STFT samples

            FLAGS = {}
            FLAGS["epoch"] = 10
            FLAGS["learning_rate"] = 0.0001
            FLAGS["beta1"] = 0.5
            FLAGS["train_size"] = np.inf
            FLAGS["batch_size"] = 64
            FLAGS["dataset"] = dataset
            FLAGS["input_fname_pattern"] = "*.jpg"
            FLAGS["checkpoint_dir"] = checkpoint 
            FLAGS["sample_dir"] = "samples_%s" %dataset           

            run_config = tf.ConfigProto()
            run_config.gpu_options.allow_growth=True

            input_height = 56
            input_width = 112
            if dataset in ['FB, CHBMIT']:
                input_height = 56
                input_width = 112
            elif dataset in ['Kaggle2014Pred']:
                input_height = 112
                input_width = 96
            elif dataset=='EpilepsiaSurf':
                input_height = 56
                input_width = 128

            with tf.Session(config=run_config) as sess:
                dcgan = DCGAN(sess=sess,
                              target=target,
                              checkpoint_dir=checkpoint,
                              dataset_dir=stft_dirs,
                              input_height=input_height,
                              input_width=input_width
                              )
                print ('Input height and width', input_height, input_width)
                dcgan.train(FLAGS)
                print ('Done training GAN')
            tf.reset_default_graph()
            # break #-- uncomment this line when training GAN  with all patients combined. i.e., only need to train once


        if build_type=='cvgan':
            makedirs(str(settings['resultdir']) + "/%s" %target)
            checkpoint = settings['ganckptdir'] + "/%s" % target # need to change dcgan/model.py pattern as well
            ictal_X, ictal_y = \
                LoadSignals(target, type='ictal', settings=settings, sph=sph).apply()
            interictal_X, interictal_y = \
                LoadSignals(target, type='interictal', settings=settings, sph=sph).apply()

            loo_folds = train_val_loo_split(ictal_X, ictal_y, interictal_X, interictal_y, 0.25)
            i_loo = 1
            for X_train, y_train, X_val, y_val, X_test, y_test in loo_folds:
                print (X_train.shape, y_train.shape,
                       X_val.shape, y_val.shape,
                       X_test.shape, y_test.shape)

                # change to channels_last
                X_train = np.transpose(X_train, (0,2,3,1))
                X_val = np.transpose(X_val, (0,2,3,1))
                X_test = np.transpose(X_test, (0,2,3,1))

                tempdir = '/mnt/data5_2T/tempdata/CHBMIT_Movidius_cv'
                makedirs(tempdir)
                tempdir += '/%s' %target
                makedirs(tempdir)
                tempdir += '/%d' %i_loo
                makedirs(tempdir)

             
                makedirs(os.path.join(settings['resultdir'],'%s' %target))
                makedirs(os.path.join(settings['resultdir'],'%s/%d' %(target,i_loo)))
                makedirs(os.path.join(settings['cnnckptdir'],'%s' %target))
                makedirs(os.path.join(settings['cnnckptdir'],'%s/%d' %(target,i_loo)))

                model = CNNGAN(
                    target,nb_classes=2,mode=build_type,
                    dataset=dataset,
                    sph=sph,
                    cache=os.path.join(settings['cnnckptdir'],'%s/%d' %(target,i_loo)),
                    checkpoint=checkpoint,
                    result_dir=os.path.join(settings['resultdir'],'%s/%d' %(target,i_loo))
                )
                model.setup(X_train.shape)
                epochs=100
                batch_size=100
                batches = int(X_train.shape[0]/batch_size)
                steps = batches*epochs
                model.fit(X_train, y_train, X_val, y_val,
                          batch_size=batch_size,steps=steps,every_n_step=batches)
                # model.load_trained_weights('./cache/Dog_1-model-5145.ckpt')
                auc = model.evaluate(X_test, y_test)
                t = '%s_%d' %(target, i_loo)
                summary[t] = auc

                # write out predictions for preictal and interictal segments
                # preictal
                X_test_p = X_test[y_test==1]
                y_test_p = model.predict_proba(X_test_p)
                filename = os.path.join(
                    str(settings['resultdir']), 'preictal_%s_%d.csv' %(target, i_loo))
                lines = []
                lines.append('preictal')
                for i in range(len(y_test_p)):
                    lines.append('%.4f' % ((y_test_p[i][1])))
                with open(filename, 'w') as f:
                    f.write('\n'.join(lines))
                print ('wrote', filename)

                # interictal
                X_test_i = X_test[y_test==0]
                y_test_i = model.predict_proba(X_test_i)
                filename = os.path.join(
                    str(settings['resultdir']), 'interictal_%s_%d.csv' %(target, i_loo))
                lines = []
                lines.append('interictal')
                for i in range(len(y_test_i)):
                    lines.append('%.4f' % ((y_test_i[i][1])))
                with open(filename, 'w') as f:
                    f.write('\n'.join(lines))
                print ('wrote', filename)

                model_infer = CNNGAN_infer(target,nb_classes=2,mode=build_type, dataset=dataset)
                model_infer.setup(X_test.shape)

                path=os.path.join(settings['resultdir'],'%s/%d' %(target,i_loo)) + '/*.meta'
                filelist = glob.glob(path)
                if len(filelist) > 0:
                    print ('Checkpoint files:', filelist)
                    fn_weights = filelist[0]
                    print (fn_weights)
                    if os.path.exists(fn_weights):
                        model_infer.load_trained_weights(fn_weights[:-5])

                i_loo += 1      

    print (summary)
    log(str(summary))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="save_STFT, dcgan, cvgan")
    parser.add_argument("--dataset", help="FB, CHBMIT or EpilepsiaSurf")
    parser.add_argument("--sph", type=int, help="0, 5, etc")
    args = parser.parse_args()
    assert args.mode in ['save_STFT','dcgan','cvgan']
    log('********************************************************************')
    log('--- START --dataset %s --mode %s ---' %(args.dataset,args.mode))
    main(dataset=args.dataset, build_type=args.mode, sph=args.sph)

