from kashgari.embeddings import BERTEmbedding
from models import CNNModel
import jieba
from tqdm import tqdm
import keras

class CNNModel(ClassificationModel):
    __architect_name__ = 'CNNModel'
    __base_hyper_parameters__ = {
        'conv1d_layer': {
            'filters': 128,
            'kernel_size': 5,
            'activation': 'relu'
        },
        'max_pool_layer': {},
        'dense_1_layer': {
            'units': 64,
            'activation': 'relu'
        },
        'activation_layer': {
            'activation': 'softmax'
        },
        'optimizer': {
            'module': 'keras.optimizers',
            'name': 'Adam',
            'params': {
                'lr': 1e-3,
                'decay': 0.0
            }
        },
        'compile_params': {
            'loss': 'categorical_crossentropy',
            # 'optimizer': 'adam',
            'metrics': ['accuracy']
        }
    }

    def build_model(self):
        base_model = self.embedding.model
        conv1d_layer = Conv1D(**self.hyper_parameters['conv1d_layer'])(base_model.output)
        max_pool_layer = GlobalMaxPooling1D(**self.hyper_parameters['max_pool_layer'])(conv1d_layer)
        dense_1_layer = Dense(**self.hyper_parameters['dense_1_layer'])(max_pool_layer)
        dense_2_layer = Dense(len(self.label2idx), **self.hyper_parameters['activation_layer'])(dense_1_layer)

        model = Model(base_model.inputs, dense_2_layer)
        optimizer = getattr(eval(self.hyper_parameters['optimizer']['module']),
                            self.hyper_parameters['optimizer']['name'])(
            **self.hyper_parameters['optimizer']['params'])
        model.compile(optimizer=optimizer, **self.hyper_parameters['compile_params'])
        self.model = model
        self.model.summary()

def read_neg_data(dataset_path):
    x_list = []
    y_list = []
    lines = open(dataset_path, 'r', encoding='utf-8').readlines()
    for line in tqdm(lines):
        line = line.strip()
        if len(line) > 1:
            label = '0'
            y_list.append(label)
            seg_text = list(jieba.cut(line))
            x_list.append(seg_text)
        else:
            continue
    return x_list, y_list

def read_pos_data(dataset_path):
    x_list = []
    y_list = []
    lines = open(dataset_path, 'r', encoding='utf-8').readlines()
    for line in tqdm(lines):
        line = line.strip()
        if len(line) > 1:
            label = '1'
            y_list.append(label)
            seg_text = list(jieba.cut(line))
            x_list.append(seg_text)
        else:
            continue
    return x_list, y_list

def concate_data(pos_x, pos_y, neg_x, neg_y):
    data_x = []
    data_y = []
    for i in range(len(pos_x)):
        data_x.append(pos_x[i])
        data_y.append(pos_y[i])
    for j in range(len(neg_x)):
        data_x.append(neg_x[j])
        data_y.append(neg_y[j])
    return data_x, data_y


def train():
    pos_data_path = 'data/pos_train.csv'
    pos_x, pos_y = read_pos_data(pos_data_path)
    print(len(pos_x))
    print(len(pos_y))
 
    neg_data_path = 'data/neg_train.csv'
    neg_x, neg_y = read_neg_data(neg_data_path)
    print(len(neg_x))
    print(len(neg_y))

    train_pos_x = pos_x[:41025]
    train_pos_y = pos_y[:41025]
    val_pos_x = pos_x[41025:52746]
    val_pos_y = pos_y[41025:52746]
    test_pos_x = pos_x[52746:]
    test_pos_y = pos_y[52746:]

    train_neg_x = neg_x[:41165]
    train_neg_y = neg_y[:41165]
    val_neg_x = neg_x[41165:52926]
    val_neg_y = neg_y[41165:52926]
    test_neg_x = neg_x[52926:]
    test_neg_y = neg_y[52926:]

    train_x, train_y = concate_data(train_pos_x, train_pos_y, train_neg_x, train_neg_y)
    val_x, val_y = concate_data(val_pos_x, val_pos_y, val_neg_x, val_neg_y)
    test_x, test_y = concate_data(test_pos_x, test_pos_y, test_neg_x, test_neg_y)

    embedding = BERTEmbedding('data/uncased_L-12_H-768_A-12', sequence_length=100)
    print('embedding_size', embedding.embedding_size)

    model = CNNModel(embedding)
    model.fit(train_x, train_y, val_x, val_y, batch_size=128, epochs=20, fit_kwargs={'callbacks': [tf_board_callback]})
    model.evaluate(test_x, test_y)
    model.save('./model/cnn_bert_model')

if __name__ == '__main__':
    train()
