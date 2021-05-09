import time
import os
import sys
from outputLog import Logger
from drawWhileRunning import draw_while_running
from adultPrivacyModels import Net,EncoderModel,TestTopModel,DecoderModel, TopModel
from dataLoader import get_data_loader,load_adult_dataset,construct_data_loader

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if __name__ == '__main__':
    file_name = 'adult'
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    outputSavePath = './test' + file_name + '_' + timestamp
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

    device = 'cuda'
    privateAttributeNumber = 1
    train_data, train_label, test_data, test_label = load_adult_dataset()
    trainLoader, testLoader = construct_data_loader(train_data, train_label, test_data, test_label, 32)
    encoder = EncoderModel()
    topModel = TestTopModel()
    model = Net(encoder,topModel,trainLoader,testLoader)
    totalEpoch = 5

    for epoch in range(totalEpoch):
        print("Epoch {}".format(epoch))
        trainAcc = model.trainModel()
        testAcc = model.testModel()
        result_file = open(os.path.join(rewardSavePath, results_name), 'a')

        result_file.write(
            str(epoch)  + '  ' + str(trainAcc) + '  '+ str(testAcc) + '  ' + '\n')
        result_file.close()

        #
        # # draw
        if epoch > 0:
            draw_while_running(rewardSavePath, results_name, rewardSavePath, str(epoch) + '_results.svg',
                               'train_vertical_model',
                               'epoch',
                               'results',
                               ['epoch', 'train_accuracy', 'test_accuracy'])

    train_feature = encoder(train_data.to(device)).detach()
    test_feature = encoder(test_data.to(device)).detach()
    top_train_loader, top_test_loader = construct_data_loader(train_feature, train_label, test_feature,
                                                              test_label, 32)
    decoder_train_feature = train_feature
    decoder_test_feature = test_feature
    test_private_label = train_data[:, privateAttributeNumber].to(device)
    train_private_label = test_data[:, privateAttributeNumber].to(device)

    decoder_train_loader, decoder_test_loader = construct_data_loader(decoder_train_feature, train_private_label,
                                                                      decoder_test_feature,
                                                                      test_private_label, 32)

    top_model = TopModel(top_train_loader, top_test_loader)
    top_model.train_model()
    top_model.test_model()


    decoder = DecoderModel(decoder_train_loader, decoder_test_loader, privateAttributeNumber)
    decoder.train_decoder()
    decoder.test_privacy()