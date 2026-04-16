close all;
clear;

% load our unsegmented images into an image store
imds = imageDatastore('cw/cw_data/images/');

% for our images, red is weed, green is crop, and black is the background
classNames = ["background" "weed" "crop"];
pixelLabelID = {[0 0 0], [255 0 0], [0 255 0]};

% load our ground truths into a pixel label data store
pxds = pixelLabelDatastore('cw/cw_data/segmentation',classNames,pixelLabelID);

% resize images to reduce computational time in training the network
% original size: 966x1296
% resized: 240x320
targetSize = [240,320];

% resize images and divide into training and test sets
inputSize = [240 320 3]; % [height width no_channels]

% divide into train and test sets (need to do validation at some point?)
% see ref #1
imgSetTrainRaw = subset(imds, 1:34);
imgSetValidateRaw = subset(imds, 35:40);
imgSetTestRaw = subset(imds, 41:50);
imgSetTrain = transform(imgSetTrainRaw,@(x) imresize(x,targetSize));
imgSetValidate = transform(imgSetValidateRaw, @(x) imresize(x, targetSize));
imgSetTest = transform(imgSetTestRaw,@(x) imresize(x,targetSize));

% do the same for the segmentation sets
segSetTrainRaw = subset(pxds, 1:34);
segSetValidateRaw = subset(pxds, 35:40);
segSetTestRaw = subset(pxds, 41:50);
segSetTrain = transform(segSetTrainRaw, @(x) {imresize(x{1}, targetSize, 'nearest')});     
segSetValidate = transform(segSetValidateRaw, @(x) {imresize(x{1}, targetSize, 'nearest')});
segSetTest  = transform(segSetTestRaw,  @(x) {imresize(x{1}, targetSize, 'nearest')});

% see ref #4 - a lot of inspiration taken from constructing the CNN as a 
% graph rather than a linear structure - allows us to implement
% unet style skipping layers
numClasses = 3;
net = dlnetwork;

encode_layers = [
    imageInputLayer(inputSize, Normalization="zscore")

    % each "layer" composed as a unet layer, 2 convs and matching BN and
    % relu. see ref #2+3
    convolution2dLayer(3, 16, 'Padding', 1, 'Name', 'conv_1int')
    batchNormalizationLayer("Name","batch_norm_1int")
    reluLayer("Name","relu_int1") 
    convolution2dLayer(3, 16, 'Padding', 1, 'Name', 'conv_1')
    batchNormalizationLayer("Name","batch_norm_1")
    reluLayer("Name","relu_1") 
    maxPooling2dLayer(2, 'Stride',2)

    convolution2dLayer(3, 32, 'Padding', 1, 'Name', 'conv_2int')
    batchNormalizationLayer("Name","batch_norm_2int")
    reluLayer("Name","relu_int2") 
    convolution2dLayer(3, 32, 'Padding', 1, 'Name', 'conv_2')
    batchNormalizationLayer("Name","batch_norm_2")
    reluLayer("Name","relu_2") 
    maxPooling2dLayer(2, 'Stride',2)
    
    convolution2dLayer(3, 64, 'Padding', 1, 'Name', 'conv_3int')
    batchNormalizationLayer("Name","batch_norm_3int")
    reluLayer("Name","relu_int3") 
    convolution2dLayer(3, 64, 'Padding', 1, 'Name', 'conv_3')
    batchNormalizationLayer("Name","batch_norm_3")
    reluLayer("Name","relu_3") 
    maxPooling2dLayer(2, 'Stride',2, "Name","pool_conn")

   % bottleneck layer
   convolution2dLayer(3, 128, 'Padding', 1, 'Name', 'bottleneck_1')
   batchNormalizationLayer("Name","batch_bn_1int")
   reluLayer("Name","relu_intbn") 
   convolution2dLayer(3, 128, 'Padding', 1, 'Name', 'bottleneck_2')
   batchNormalizationLayer("Name","batch_bn_1")
   reluLayer("Name","relu_bn")

   % dropout to reduce overfitting - noticed in ref #2 and elaborated on in
   % ref #8
   dropoutLayer(0.3, "Name","dropout")
]; 

net = addLayers(net, encode_layers);

decode_layers = [
    % corresponding decode layers act as upsampling, matched again with BN
    % and relu layers. we also add some concatenation layers to allow us to
    % create the unet architecture
    transposedConv2dLayer(4, 64, 'Stride',2, 'Cropping', 1, 'Name', 'conv_4')
    batchNormalizationLayer
    reluLayer("Name","relu_4") 
    depthConcatenationLayer(2, 'Name', 'concat1')
    convolution2dLayer(3, 64, 'Padding', 1)
    batchNormalizationLayer
    reluLayer("Name","relu_5") 

    transposedConv2dLayer(4, 32, 'Stride',2, 'Cropping', 1, 'Name', 'conv_5')
    batchNormalizationLayer
    reluLayer("Name","relu_6")
    depthConcatenationLayer(2, 'Name', 'concat2')
    convolution2dLayer(3, 32, 'Padding', 1)
    batchNormalizationLayer
    reluLayer("Name","relu_7") 
    
    transposedConv2dLayer(4, 16, 'Stride',2, 'Cropping', 1, 'Name', 'conv_6')
    batchNormalizationLayer
    reluLayer
    depthConcatenationLayer(2, 'Name', 'concat3')
    convolution2dLayer(3, 16, 'Padding', 1)
    batchNormalizationLayer("Name","batch_norm_6")
    reluLayer

    % final classification layer after all the heavy computing
    convolution2dLayer(1, numClasses, 'Name', 'conv_7');
    softmaxLayer
];

% connect the layers together
net = addLayers(net, decode_layers);
net = connectLayers(net, "dropout", "conv_4");
net = connectLayers(net, "relu_1", "concat3/in2");
net = connectLayers(net, "relu_2", "concat2/in2");
net = connectLayers(net, "relu_3", "concat1/in2");

trainingData = combine(imgSetTrain, segSetTrain);
validationData = combine(imgSetValidate, segSetValidate);
trainNewModel = true;  

% training hyperparameters, working these out was a pain
% contention between SGDM with 1e-2 or adam with 1e-3.
opts = trainingOptions('adam', ...
   'InitialLearnRate',1e-3, ...
   'MaxEpochs',50,...
   'MiniBatchSize',4, ...
   'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10, ...
    'LearnRateDropFactor',0.5, ...
    'ValidationData',validationData,...
    'ValidationFrequency',8,...
    'ValidationPatience',8 ...
   );

% see ref #7 - made the choice of multiple loss functions, need to optimise
% how they are used/defined for readability - this was haphazardly put
% together and I'm not sure if its right
function customLossFunction = diceAndCE(predictions, truths, frequencies)
    % frequencies - used to calculate weighting
    % predictions - our softmax normalised predictions
    % truths - the actual classification of the pixels

    % calculate the weights
    classWeights = 1 ./ sqrt(frequencies);
    classWeights = reshape(classWeights, 1, 1, []);

    % calculate dice loss - ref #9
    dice = 1 - mean(generalizedDice(predictions, truths), "all");

    % calculate CE loss
    ce = -mean(classWeights .* truths .* log(predictions + 1e-8),"all");

    % add them together
    customLossFunction = 0.5*dice + 0.5*ce;
end



% model training! if we want a new model, train it, otherwise we can use
% for evaluation of previously trained models.
if trainNewModel                                                                                                                                                                                      
  tbl          = countEachLabel(segSetTrainRaw);                                                                                                                                                           
  frequency    = tbl.PixelCount / sum(tbl.PixelCount);   
  net = trainnet(trainingData, net, @(predictions, truths) diceAndCE(predictions, truths, frequency), opts);
  save('segmentnet_base', 'net');                                                                                                                                                                   
else                                                                                                                                                                                                  
  load('segmentnet_base', 'net');                                                                                                                                                                   
end 

% perform the segmentation!
pxdsResultsRaw = semanticseg(imgSetTest, net, 'WriteLocation', pwd);
pxdsResults = transform(pxdsResultsRaw, @(x) {imresize(x{1}, [966 1296], 'nearest')});
pxdsResults = transform(pxdsResults, @(x) {renamecats(x{1}, classNames)}); 

% evaluate the segmentation.
metrics = evaluateSemanticSegmentation(pxdsResults, segSetTestRaw);

figure;
cm = confusionchart(metrics.ConfusionMatrix.Variables, classNames, Normalization="row-normalized");
cm.Title = "Normalised Confusion Matrix";

% visualise some results
testImg = readimage(imgSetTestRaw, 3);
predSeg = readimage(pxdsResultsRaw, 3);
predSeg = imresize(predSeg, [966 1296], 'nearest');
figure;
imshow(labeloverlay(testImg, predSeg));
title('Overlay image!');

figure;
imshow(testImg);
title('Test Image!');


