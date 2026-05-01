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
inputSize = [240 320 3];

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

   % bottleneck layer to sit at bottom of unet
   convolution2dLayer(3, 256, 'Padding', 1, 'Name', 'bottleneck_1')
   batchNormalizationLayer("Name","batch_bn_1int")
   reluLayer("Name","relu_intbn") 
   convolution2dLayer(3, 256, 'Padding', 1, 'Name', 'bottleneck_2')
   batchNormalizationLayer("Name","batch_bn_1")
   reluLayer("Name","relu_bn")

   % dropout to reduce overfitting - noticed in ref #2 and elaborated on in
   % ref #8
   dropoutLayer(0.3, "Name","dropout")
]; 

% see dlnetwork addLayers function
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

% formulate training and validation data
% for improved network, this is where the augmentation takes place
trainingData = transform(combine(imgSetTrain, segSetTrain), @augmentData);
validationData = combine(imgSetValidate, segSetValidate);

% work out how many elements are in new training data set
% numel(originalTrainingData.readall)
% nnumel(trainingData.readall)

% training hyperparameters, working these out was a pain
% contention between SGDM with 1e-2 or adam with 1e-3.
opts = trainingOptions('adam', ...
   'InitialLearnRate',1e-3, ...
   'MaxEpochs',100,...
   'MiniBatchSize',4, ...
   'LearnRateSchedule','piecewise',...
   'LearnRateDropPeriod',10, ...
   'LearnRateDropFactor',0.5, ...
   'ValidationData',validationData,...
   'ValidationFrequency',4,...
   'ValidationPatience',30, ...
   'Plots','training-progress', ...
   'Metrics','accuracy',...
   'Shuffle','every-epoch'...
   );

% change this to true when you want to train a new model (est. 3-4 minutes)
trainNewModel = true;  

% model training! if we want a new model, train it, otherwise we can use
% for evaluation of previously trained models.
if trainNewModel                                                                                                                                                                                      
  tbl          = countEachLabel(segSetTrainRaw);                                                                                                                                                           
  frequency    = tbl.PixelCount / sum(tbl.PixelCount);   
  net = trainnet(trainingData, net, @(predictions, truths) diceAndFocal(predictions, truths, frequency), opts);
  
  save('segmentnet_imp', 'net');   
  netImp = net;
  load('segmentnet_base', 'net');
  netBase = net;
else                                                                                                                                                                                                  
  load('segmentnet_imp', 'net');
  netImp = net;
  load('segmentnet_base', 'net');
  netBase = net;
end 

% perform the segmentation, for each image segment, very slight upscale,
% then write labelled image to segmentationResults directory
outputDir = fullfile(pwd, 'segmentationImprovedResults');
baseDir = fullfile(pwd, 'segmentationResults');
if ~exist(outputDir, 'dir'); mkdir(outputDir); end

i = 1;
while hasdata(imgSetTest)
    img = read(imgSetTest);                          
    predSmall = semanticseg(img, netImp);              
    predFull = imresize(predSmall, [966 1296], 'nearest');
    imwrite(label2rgb(uint8(predFull), [0 0 0; 1 0 0; 0 1 0]), fullfile(outputDir, sprintf('prediction_%02d.png', i)));

    i = i + 1;
end

% read results back in after they've been generated
pxdsResults = pixelLabelDatastore(outputDir, classNames, {[0 0 0], [255 0 0], [0 255 0]});
pxdsBaseResults = pixelLabelDatastore(baseDir, classNames, {[0 0 0], [255 0 0], [0 255 0]});

% evaluate the segmentation.
metrics = evaluateSemanticSegmentation(pxdsResults, segSetTestRaw);
metrics2 = evaluateSemanticSegmentation(pxdsBaseResults, segSetTestRaw);
perClassMetrics = metrics.ClassMetrics;
disp('Improved Network: Class Metrics');
disp('-------------------------------------');
disp(perClassMetrics);
perClassMetrics2 = metrics2.ClassMetrics;
disp('Base Network: Class Metrics');
disp('-------------------------------------');
disp(perClassMetrics2);

figure;
tiledlayout(1,2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
cm = confusionchart(metrics.ConfusionMatrix.Variables, classNames, Normalization="row-normalized");
cm.Title = "Normalised Confusion Matrix";
nexttile;
cm2 = confusionchart(metrics2.ConfusionMatrix.Variables, classNames, Normalization="row-normalized");
cm2.Title = "Normalised Base Confusion Matrix";

figure;
numImages = 5;
% instead of using typical subplots used tiledlayout - easier
tiledlayout(numImages, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:numImages
    nexttile;
    img = readimage(imgSetTestRaw, i);
    imshow(img);
    if i == 1; title('Original Image'); end;
    nexttile;
    predSeg = readimage(pxdsResults, i);
    %predSeg = imresize(predSeg, [966 1296], 'nearest');
    imshow(labeloverlay(img, predSeg));
    if i == 1; title('Improved Segmentation'); end;
    nexttile;
    predSeg = readimage(pxdsBaseResults, i);
    imshow(labeloverlay(img, predSeg));
    segTruth = readimage(segSetTestRaw, i);
    if i == 1; title('Base Segmentation'); end;
    nexttile;
    imshow(labeloverlay(img, segTruth));
    if i == 1; title('Truth Segmentation'); end;
end

% HELPER FUNCTIONS

% see ref #7 - made the choice of multiple loss functions, need to optimise
% how they are used/defined for readability - this was haphazardly put
% together and I'm not sure if its right
function customLossFunction = diceAndFocal(predictions, truths, frequencies)
    % frequencies - used to calculate weighting
    % predictions - our softmax normalised predictions
    % truths - the actual classification of the pixels

    % calculate the weights
    %classWeights = 1 ./ sqrt(frequencies);
    classWeights = median(frequencies) ./ frequencies;
    classWeights = reshape(classWeights, 1, 1, []);

    % calculate dice loss - ref #9
    dice = 1 - mean(generalizedDice(predictions, truths), "all");
    
    % focal loss
    gamma = 1.5;
    pt = sum(predictions .* truths, 3);  
    alpha_t = sum(classWeights .* truths, 3);
    focal = -mean(alpha_t .* (1 - pt).^gamma .* log(pt + 1e-8), "all");

    % add them together
    customLossFunction = 0.6*dice + 0.4*focal;
end

% initial augmentation boilerplate - not used yet
function output = augmentData(images)
    
    image = images{1};
    segmentation = images{2};
    
    % left right flip
    if rand < 0.5
        image = fliplr(image);
        segmentation = fliplr(segmentation);
    end

    % up down flip
    if rand < 0.5
        image = flipud(image);
        segmentation = flipud(segmentation);
    end

    img = im2single(image);
    
    % brightness
    if rand < 0.4
        img = img + (rand*0.2 - 0.1);  
    end
    
    % contrast
    if rand < 0.3
        img = (img - 0.5) * (0.8 + rand*0.4) + 0.5; 
    end
    
    img = im2uint8(min(max(img, 0), 1));
    image = img;
    output = {image, segmentation};
end