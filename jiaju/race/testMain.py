import time
import os
import sys
from outputLog import Logger
from drawWhileRunning import draw_while_running
from privacyModels import Net,EncoderModel,TopModel,Decoder,CludTask
from dataLoader import load_adult_dataset,construct_data_loader,construct_data_loader1
import os
import torch
import datetime
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
if __name__ == '__main__':
    device = 'cuda'

    featureDim = 256  # feature Size

    whichprivacyLoss = 2
    batchSize = 256
    encoderEpoch = 20
    topModelEpcoh = 20
    decoderEpoch = 1000

    withoutPA = 0#
    lam1 = 0.5
    printLoss = 0
    printLossDetal = 0

    activateLoss = 0


    flag = 1
    if flag:
        file_name = 'race_m{}_batchSize{}'.format(whichprivacyLoss,batchSize)
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



    print("which privacy loss:", whichprivacyLoss)
    print("batchSize:",batchSize)



    train_data,train_label, test_data, test_label, train_privact_att,test_privact_att  = load_adult_dataset()
    # train_data,train_privact_att, test_data, test_privact_att, train_label,test_label  = load_adult_dataset()

    trainLoader, testLoader = construct_data_loader(train_data, train_label, test_data, test_label, train_privact_att, test_privact_att, batchSize)

    encoder = EncoderModel()
    testTopModel = TopModel()
    model = Net(encoder,testTopModel,trainLoader,testLoader,whichprivacyLoss,featureDim,device)


    start = datetime.datetime.now()

    for epoch in range(encoderEpoch):
        print("Epoch {}:".format(epoch))
        trainAcc = model.trainModel(lam= lam1,pd1=printLoss,pd2=printLossDetal,activateLoss=activateLoss)
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

    torch.save(encoder,"encoder_m{}_batchSize{}".format(whichprivacyLoss,batchSize))
    # train_feature = encoder(train_data.to(device)).detach()
    # test_feature = encoder(test_data.to(device)).detach()
    train_feature = train_data.to(device)
    test_feature = test_data.to(device)
    top_train_loader, top_test_loader = construct_data_loader1(train_feature, train_label, test_feature,
                                                              test_label, batchSize)


    decoder_train_feature = train_feature
    decoder_test_feature = test_feature


    decoder_train_loader, decoder_test_loader = construct_data_loader1(decoder_train_feature, train_privact_att,
                                                                      decoder_test_feature,
                                                                      test_privact_att, batchSize)

    # start = datetime.datetime.now()
    # MIfeature = train_feature.to(device)
    # MIlabel = train_private_label.to(device)
    # MutInf = MutualInformation(1, device)
    # MutInf.forward(train_feature, train_private_label)
    #
    # muinfTime = datetime.datetime.now() - start
    # print("MutInf Time:",muinfTime)
    top_model = CludTask(encoder,top_train_loader, top_test_loader)
    start = datetime.datetime.now()
    top_model.train_model(topModelEpcoh)
    top_model.test_model()

    topModelTrainingTime = datetime.datetime.now() - start

    decoder = Decoder(encoder,decoder_train_loader, decoder_test_loader)
    start = datetime.datetime.now()
    decoder.train_decoder(decoderEpoch)
    decoder.test_decoder()
    decoderTrainingTime = datetime.datetime.now() - start
    print("encoder training time:",encoderTrainingTime)
    print("decoder training time:", decoderTrainingTime)
    print("topModel training time:", topModelTrainingTime)