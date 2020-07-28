import os
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import ktrain
from ktrain import text


def read_document(filepath):
    arq = open(filepath, 'r')
    doc = arq.readlines()
    arq.close()
    documents = list(map(str.rstrip, doc))
    return documents


def write_output(y, pred, filepath):
    with open(filepath, 'w') as output:
        for doc_iter in range(0, len(y)):
            output.write('{iter} CLASS={y} CLASS={pred}:1.0\n'.format(iter=doc_iter + 1,
                                                                      y=y[doc_iter],
                                                                      pred=pred[doc_iter].split('_')[1]))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset',
                        action='store',
                        type=str,
                        dest='dataset',
                        help='--dataset [dataset name]')
    parser.add_argument('-f', '--fold',
                        action='store',
                        type=str,
                        dest='fold',
                        help='--fold [fold id]')
    parser.add_argument('-t', '--train',
                        action='store',
                        type=str,
                        dest='train',
                        help='--train [train file name]')
    parser.add_argument('-l', '--test',
                        action='store',
                        type=str,
                        dest='test',
                        help='--test [test file name]')
    parser.add_argument('-c', '--class',
                        action='store',
                        type=str,
                        dest='classes',
                        help='--class [test labels file name]')
    parser.add_argument('-o', '--output',
                        action='store',
                        type=str,
                        dest='out',
                        help='--output [output file name]')
    args = parser.parse_args()

    (x_train, y_train), (x_val, y_val), preproc = text.texts_from_csv(train_filepath=args.train,
                                                                      text_column='text',
                                                                      maxlen=150,
                                                                      preprocess_mode='bert',
                                                                      label_columns=['class_0', 'class_1'])

    # (x_train, y_train), (x_test, y_test), preproc = text.texts_from_csv(train_filepath=args.train, 
    #                                                                 val_filepath=args.test,
    #                                                                 text_column='text',
    #                                                                 preprocess_mode='bert',
    #                                                                 label_columns=['class_0','class_1','class_2'])

    time_exec = dict()
    x_test = read_document(filepath=args.test)
    start_time = time.time()
    model = text.text_classifier('bert', train_data=(x_train, y_train), preproc=preproc)
    end_time = time.time()
    time_exec['learning'] = end_time - start_time

    start_time = time.time()
    learner = ktrain.get_learner(model, train_data=(x_train, y_train), val_data=(x_val, y_val), batch_size=32)
    end_time = time.time()
    time_exec['get_learner'] = end_time - start_time

    start_time = time.time()
    learner.autofit(lr=2e-5, epochs=None)
    end_time = time.time()
    time_exec['auto_fit'] = end_time - start_time

    start_time = time.time()
    learner.validate(val_data=(x_val, y_val))
    end_time = time.time()
    time_exec['validate'] = end_time - start_time

    start_time = time.time()
    predictor = ktrain.get_predictor(learner.model, preproc)
    pred_test = predictor.predict(x_test)
    y_test = read_document(filepath=args.classes)
    write_output(y=y_test, pred=pred_test, filepath=args.out)
    end_time = time.time()
    time_exec['predict'] = end_time - start_time

    with open(f'time_{args.dataset}_{args.fold}.txt', 'w') as output:
        for key, value in time_exec.items():
            output.write(f'{key} {value}\n')


if __name__ == '__main__':
    run()
