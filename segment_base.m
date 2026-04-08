close all;
clear;

% load our unsegmented images into an image store
imds = imageDatastore('cw/cw_data/images/');

% for our images, red is weed, green is crop, and black is the background
classNames = ["background" "weed" "crop"];
pixelLabelID = {[0 0 0], [255 0 0], [0 255 0]};

% load our ground truths into a pixel label data store
pxds = pixelLabelDatastore('cw/cw_data/segmentation',classNames,pixelLabelID);

% boiler-plate initial comparison for understanding
% comparison = labeloverlay(readimage(imds,1), readimage(pxds, 1));
% figure;
% imshow(comparison);

% resize images to reduce computational time
% original size: 966x1296
% resized: 320x432
targetSize = [320,432];

% resize images and divide into training and test sets
inputSize = [320 432 3]; % [height width no_channels]

imgSetTrainRaw = subset(imds, 1:40);
imgSetTestRaw = subset(imds, 41:50);
imgSetTrain = transform(imgSetTrainRaw,@(x) imresize(x,targetSize));
imgSetTest = transform(imgSetTestRaw,@(x) imresize(x,targetSize));

segSetTrainRaw = subset(pxds, 1:40);
segSetTestRaw = subset(pxds, 41:50);
segSetTrain = transform(segSetTrainRaw, @(x) {imresize(x{1}, targetSize, 'nearest')});                                                    
segSetTest  = transform(segSetTestRaw,  @(x) {imresize(x{1}, targetSize, 'nearest')});

numClasses = 3;
net = dlnetwork;

% apply weighting
tbl          = countEachLabel(segSetTrainRaw);                                                                                                                                                           
frequency    = tbl.PixelCount / sum(tbl.PixelCount);                                                                                                                                                  
classWeights = median(frequency) ./ frequency;  

encode_layers = [
    imageInputLayer(inputSize)

    convolution2dLayer(3, 16, 'Padding', 1, 'Name', 'conv_1')
    batchNormalizationLayer("Name","batch_norm_1")
    reluLayer("Name","relu_1") 
    maxPooling2dLayer(2, 'Stride',2)

    convolution2dLayer(3, 32, 'Padding', 1, 'Name', 'conv_2')
    batchNormalizationLayer("Name","batch_norm_2")
    reluLayer("Name","relu_2") 
    maxPooling2dLayer(2, 'Stride',2)

    convolution2dLayer(3, 64, 'Padding', 1, 'Name', 'conv_3')
    batchNormalizationLayer("Name","batch_norm_3")
    reluLayer("Name","relu_3") 
    maxPooling2dLayer(2, 'Stride',2, "Name","pool_conn")

]
net = addLayers(net, encode_layers);

decode_layers = [
    transposedConv2dLayer(4, 64, 'Stride',2, 'Cropping', 1, 'Name', 'conv_4')
        depthConcatenationLayer(2, 'Name', 'concat1')

    batchNormalizationLayer("Name","batch_norm_4")
    reluLayer("Name","relu_4") 

    transposedConv2dLayer(4, 32, 'Stride',2, 'Cropping', 1, 'Name', 'conv_5')
        depthConcatenationLayer(2, 'Name', 'concat2')

    batchNormalizationLayer("Name","batch_norm_5")
    reluLayer("Name","relu_5") 

    transposedConv2dLayer(4, 16, 'Stride',2, 'Cropping', 1, 'Name', 'conv_6')
    depthConcatenationLayer(2, 'Name', 'concat3')

    batchNormalizationLayer("Name","batch_norm_6")
    reluLayer("Name","relu_6") 

    convolution2dLayer(1, numClasses, 'Name', 'conv_7');
    softmaxLayer()
    %pixelClassificationLayer('Classes', classNames,'ClassWeights', classWeights )

];

net = addLayers(net, decode_layers);
net = connectLayers(net, "pool_conn", "conv_4");
net = connectLayers(net, "relu_1", "concat3/in2");
net = connectLayers(net, "relu_2", "concat2/in2");
net = connectLayers(net, "relu_3", "concat1/in2");

figure;
plot(net);


opts = trainingOptions('sgdm', ...
   'InitialLearnRate',1e-2, ...
   'MaxEpochs',10,...
   'MiniBatchSize',2, ...
   'LearnRateSchedule','piecewise',...
   'LearnRateDropPeriod',6, ...
   'LearnRateDropFactor',0.1 ...
   ...
   );

trainingData = combine(imgSetTrain, segSetTrain);
trainNewModel = false;                                                                                                                           
                                                                                                                                                                                                        
if trainNewModel                                                                                                                                                                                      
  %net = trainNetwork(trainingData, layers, opts);  
  net = trainnet(trainingData, net,'crossentropy', opts);
  save('segmentnet_base', 'net');                                                                                                                                                                   
else                                                                                                                                                                                                  
  load('segmentnet_base', 'net');                                                                                                                                                                   
end 

pxdsResultsRaw = semanticseg(imgSetTest, net, 'WriteLocation', pwd);
pxdsResults = transform(pxdsResultsRaw, @(x) {imresize(x{1}, [966 1296], 'nearest')});
pxdsResults = transform(pxdsResults, @(x) {renamecats(x{1}, classNames)}); 
metrics = evaluateSemanticSegmentation(pxdsResults, segSetTestRaw);

testImg = readimage(imgSetTestRaw, 1);
predSeg = readimage(pxdsResultsRaw, 1);
predSeg = imresize(predSeg, [966 1296], 'nearest');
figure;
imshow(labeloverlay(testImg, predSeg));
title('Overlay image!');

figure;
imshow(testImg);
title('Test Image!');

figure;
cm = confusionchart(metrics.ConfusionMatrix.Variables, classNames, Normalization="row-normalized");
cm.Title = "Normalised Confusion Matrix";
