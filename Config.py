


class Config(object):
    def __init__(self, word_embedding_dimension=100, word_num=20000,
                 epoch=2, sentence_max_size=32, cuda=False,
                 learning_rate=0.01, batch_size=1, seed=1,
                 dropout=0.1, bilstm_hidden_dim=50, bilstm_num_layers=1):
        self.word_embedding_dimension = word_embedding_dimension     # 词向量的维度
        self.word_num = word_num
        self.epoch = epoch                                           # 遍历样本次数
        self.sentence_max_size = sentence_max_size                   # 句子长度                                 # 分类标签个数
        self.lr = learning_rate
        self.batch_size = batch_size
        self.seed = seed
        self.cuda = cuda
        self.dropout = dropout
        self.dataset_path = './data/...'
        self.sentiment_list = ['positive', 'neutral', 'negative']
        self.sar_list = ['neutral', 'anger', 'disgust', 'fear', 'sad']
        self.sen_num = len(self.sentiment_list)
        self.sar_num = len(self.sarcasm_list)
        self.bert_name = 'bert-base-uncased'