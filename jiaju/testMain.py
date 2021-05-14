import time
import os
import sys
from outputLog import Logger
from drawWhileRunning import draw_while_running
from adultPrivacyModels import Net,EncoderModel,TestTopModel,DecoderModel, TopModel
from dataLoader import load_adult_dataset,construct_data_loader
from lossFunction import MutualInformation
import torch
from tqdm import tqdm
import datetime

if __name__ == '__main__':
    device = 'cuda'
    privateAttributeNumber = 7
    type_num = [9, 4, 9, 7, 15, 6, 5, 2]
    whichprivacyLoss = 2
    batchSize = 256
    encoderEpoch = 10
    topModelEpcoh = 30
    decoderEpoch = 30
    lam = 1.0E-11
    withoutPA = 0

    flag = 1
    if flag:
        file_name = 'adult_m{}_p{}_lam{}_'.format(whichprivacyLoss, privateAttributeNumber, lam)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        outputSavePath = './' + file_name + '_' + timestamp
        if not os.path.exists(outputSavePath):
            os.mkdir(outputSavePath)
        logSavePath = outputSavePath + '/log'
        if not os.path.exists(logSavePath):
            os.mkdir(logSavePath)
        sys.stdout = Logger(os.path.join(logSavePath, "output.txt"), sys.stdout)
        sys.stderr = Logger(os.path.join(logSavePath, "error.txt"), sys.stderr)
        rewardSavePath = outputSavePath + '/saveReward'
        if not os.path.exists(rewardSavePath):
            os.mkdir(rewardSavePath)
        results_name = 'results_log.txt'
        accuracy_file = open(os.path.join(rewardSavePath, results_name), 'w')


    print("private Attribute Number:", privateAttributeNumber, " ", type_num[privateAttributeNumber])
    print("which privacy loss:", whichprivacyLoss)
    print("lam:", lam)



    train_data, train_label, test_data, test_label = load_adult_dataset(privateAttributeNumber,withoutPA)

    trainLoader, testLoader = construct_data_loader(train_data, train_label, test_data, test_label, batchSize)

    encoder = EncoderModel()
    topModel = TestTopModel()
    model = Net(encoder,topModel,trainLoader,testLoader,whichprivacyLoss)


    start = datetime.datetime.now()
    range(encoderEpoch)
    for epoch in range(encoderEpoch):
        print("Epoch {}:".format(epoch))
        trainAcc = model.trainModel(lam,privateAttributeNumber)
        testAcc = model.testModel()
        if flag:
            result_file = open(os.path.join(rewardSavePath, results_name), 'a')

            result_file.write(
                str(epoch) + '  ' + str(trainAcc) + '  ' + str(testAcc) + '  ' + '\n')
            result_file.close()

            # draw
            if epoch > 0:
                draw_while_running(rewardSavePath, results_name, rewardSavePath, str(epoch) + '_results.svg',
                                   'train_vertical_model',
                                   'epoch',
                                   'results',
                                   ['epoch', 'train_accuracy', 'test_accuracy'])

    encoderTrainingTime = datetime.datetime.now()-start


    train_feature = encoder(train_data.to(device)).detach()
    test_feature = encoder(test_data.to(device)).detach()
    top_train_loader, top_test_loader = construct_data_loader(train_feature, train_label, test_feature,
                                                              test_label, batchSize)


    decoder_train_feature = train_feature
    decoder_test_feature = test_feature
    train_data111, _, test_data111, _ = load_adult_dataset(privateAttributeNumber, 0)
    train_private_label = train_data111[:, privateAttributeNumber].to(device)
    test_private_label = test_data111[:, privateAttributeNumber].to(device)
    decoder_train_loader, decoder_test_loader = construct_data_loader(decoder_train_feature, train_private_label,
                                                                      decoder_test_feature,
                                                                      test_private_label, batchSize)

    # start = datetime.datetime.now()
    # MIfeature = train_feature.to(device)
    # MIlabel = train_private_label.to(device)
    # MutInf = MutualInformation(1, device)
    # MutInf.forward(train_feature, train_private_label)
    #
    # muinfTime = datetime.datetime.now() - start
    # print("MutInf Time:",muinfTime)
    top_model = TopModel(top_train_loader, top_test_loader)
    start = datetime.datetime.now()
    top_model.train_model(topModelEpcoh)
    top_model.test_model()

    topModelTrainingTime = datetime.datetime.now() - start

    decoder = DecoderModel(decoder_train_loader, decoder_test_loader, privateAttributeNumber,type_num)
    start = datetime.datetime.now()
    decoder.train_decoder(decoderEpoch)
    decoder.test_decoder()
    decoderTrainingTime = datetime.datetime.now() - start
    print("encoder training time:",encoderTrainingTime)
    print("decoder training time:", decoderTrainingTime)
    print("topModel training time:", topModelTrainingTime)