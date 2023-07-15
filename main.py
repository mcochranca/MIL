import Config
from Model import Model
import torch
import random
import numpy as np
from tqdm import trange, tqdm
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader


def set_seed(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)


def evaluate(config, model, data_loader, model_name, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in data_loader:
                if config.cuda and torch.cuda.is_available():
                    texts = texts.cuda()
                    masks = masks.cuda()
                    images = images.cuda()
                    videos = videos.cuda()
                    sen_labels = sen_labels.cuda()
                    sar_labels = sar_labels.cuda()
                out = model.forward(texts, masks, images,videos)

                loss1 = F.cross_entropy(out, sen_labels)
                labels = sen_labels.data.cpu().numpy()
                loss2 = F.cross_entropy(out, sar_labels)
                labels = sar_labels.data.cpu().numpy()

            loss_total = loss1+loss2

            predic = torch.max(out.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    report1 = metrics.classification_report(labels_all, predict_all, target_names=config.sentiment_list,
                                                   digits=4)
    report2 = metrics.classification_report(labels_all, predict_all, target_names=config.sarcasm_list, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)

    return acc, loss_total


def train(config, embedding, model_name, ):
    train_set = dataset(config=config, embedding=embedding, model_name=model_name, mode='train')
    dev_set = dataset(config=config, embedding=embedding, model_name=model_name, mode='dev')

    train_loader = DataLoader(train_set, shuffle=True, batch_size=config.batch_size, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_set, shuffle=False, batch_size=config.batch_size, num_workers=4, pin_memory=True)

    model = Model(config, mode)

    if config.cuda and torch.cuda.is_available():
        model.cuda()
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True,
                                                           threshold=0.0001, threshold_mode='rel', cooldown=2, min_lr=0,
                                                           eps=1e-08)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 50, 70, 90], gamma=0.1, last_epoch=-1)

    # for epoch in trange(config.epoch, desc="Epoch"):
    for epoch in range(config.epoch):
        print('epoch:', epoch)
        step = 0
        # for i, batch in enumerate(tqdm(train_loader, desc="batch_nums")):
        for i, batch in enumerate(train_loader):
            step += 1
            model.zero_grad()
                if config.cuda and torch.cuda.is_available():
                    texts = texts.cuda()
                    images = images.cuda()
                    videos = videos.cuda()
                    sen_labels = sen_labels.cuda()
                    sar_labels = sar_labels.cuda()
                out = model.forward(texts, images,videos)
            else:
                texts, masks, images, sen_labels, sar_labels= batch
                if config.cuda and torch.cuda.is_available():
                    texts = texts.cuda()
                    masks = masks.cuda()
                    images = images.cuda()
                    sen_labels = sen_labels.cuda()
                    sar_labels = sar_labels.cuda()
                out = model.forward(texts, masks, images,videos)
                loss1 = criterion(out, sen_labels)
                loss2 = criterion(out, sar_labels)
            loss=loss1+loss2

            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                dev_acc, dev_loss = evaluate(config, model, dev_loader, model_name=model_name)
                print('\ndev_acc: {},'.format(dev_acc), 'dev_loss: {}'.format(dev_loss))
                model.train()

        scheduler.step(loss)
        # scheduler.step()

    return model


def test(config, embedding, model, model_name,mode):
    model.eval()
    # test_loader = read_data(config=config, embedding=embedding, attr='test')
    test_set = dataset(config=config, embedding=embedding, model_name=model_name, mode='test')
    test_loader = DataLoader(test_set, shuffle=False, batch_size=config.batch_size, num_workers=4, pin_memory=True)
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_loader, model_name=model_name,
                                                                test=True, mode=mode)
    # msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    # print(msg.format(test_loss, test_acc))
    # print("Precision, Recall and F1-Score...")
    # print(test_report)
    # print("Confusion Matrix...")
    # print(test_confusion)


def main():
    # Create the configuration
    config = Config(sentence_max_size=50,
                    batch_size=32,
                    seed=9,
                    word_num=11000,
                    learning_rate=1e-5,
                    cuda=True,
                    epoch=100,
                    dropout=0.5)

    model_names = 'Model'
    mode = 'sentiment'
    set_seed(config)
    word2idx, embedding = text_process(config)
    model = train(config=config, embedding=embedding, model_name=model_name, mode=mode)
    test(config=config, embedding=embedding, model=model, model_name=model_name, mode=mode)
    # torch.save(model, 'BertRes_des.pth')


if __name__ == '__main__':
    main()