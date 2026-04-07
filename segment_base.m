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

% apply weighting to the images - there's a lot of background!
tbl          = countEachLabel(segSetTrainRaw);                                                                                                                                                           
frequency    = tbl.PixelCount / sum(tbl.PixelCount);                                                                                                                                                  
classWeights = median(frequency) ./ frequency;  

layers = [
    imageInputLayer(inputSize)

    convolution2dLayer(3, 16, 'Padding', 1)
    reluLayer() 
    batchNormalizationLayer()
    maxPooling2dLayer(2, 'Stride',2)

    convolution2dLayer(3, 32, 'Padding', 1)
    reluLayer() 
    batchNormalizationLayer()
    maxPooling2dLayer(2, 'Stride',2)

    convolution2dLayer(3, 64, 'Padding', 1)
    reluLayer()
    batchNormalizationLayer()
    maxPooling2dLayer(2, 'Stride',2)

    transposedConv2dLayer(4, 64, 'Stride',2, 'Cropping', 1)
    reluLayer()
    batchNormalizationLayer()

    transposedConv2dLayer(4, 32, 'Stride',2, 'Cropping', 1)
    reluLayer()
    batchNormalizationLayer()

    transposedConv2dLayer(4, 16, 'Stride',2, 'Cropping', 1)
    reluLayer()
    batchNormalizationLayer()

    convolution2dLayer(1, numClasses);
    softmaxLayer()
    pixelClassificationLayer('Classes', classNames,'ClassWeights', classWeights )

]

opts = trainingOptions('sgdm', ...
   'InitialLearnRate',1e-2, ...
   'MaxEpochs',20,...
   'MiniBatchSize',2, ...
   'LearnRateSchedule','piecewise',...
   'LearnRateDropPeriod',6, ...
   'LearnRateDropFactor',0.1 ...
   ...
   );

trainingData = combine(imgSetTrain, segSetTrain);
trainNewModel = true;                                                                                                                           
                                                                                                                                                                                                        
  if trainNewModel                                                                                                                                                                                      
      net = trainNetwork(trainingData, layers, opts);                                                                                                                                                   
      save('segmentnet_base', 'net');                                                                                                                                                                   
  else                                                                                                                                                                                                  
      load('segmentnet_base', 'net');                                                                                                                                                                   
  end 

pxdsResultsRaw = semanticseg(imgSetTest, net, 'WriteLocation', pwd);
pxdsResults = transform(pxdsResultsRaw, @(x) {imresize(x{1}, [966 1296], 'nearest')});
metrics = evaluateSemanticSegmentation(pxdsResults, segSetTestRaw);

testImg = readimage(imgSetTestRaw, 7);
predSeg = readimage(pxdsResultsRaw, 7);
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
