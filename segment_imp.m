close all;
clear;

% load our unsegmented images into an image store
imds = imageDatastore('cw_data/images/');

% for our images, red is weed, green is crop, and black is the background
classNames = ["background" "weed" "crop"];
pixelLabelID = {[0 0 0], [255 0 0], [0 255 0]};

% load our ground truths into a pixel label data store
pxds = pixelLabelDatastore('cw_data/segmentation',classNames,pixelLabelID);

% resize images to reduce computational time in training the network
% original size: 966x1296
% resized: 240x320 (1/4 of the size)
targetSize = [240,320];

% set the rng seed to be the same each time (used in tandem with the
% segment_base). use 42 as it's pretty standard across randomising models.
rng(42);
randomIndexes = randperm(50);

% resize images and divide into training, test and val sets.
imgSetTrainRaw = subset(imds, randomIndexes(1:34));
imgSetValidateRaw = subset(imds, randomIndexes(35:40));
imgSetTestRaw = subset(imds, randomIndexes(41:50));
imgSetTrain = transform(imgSetTrainRaw,@(x) imresize(x,targetSize));
imgSetValidate = transform(imgSetValidateRaw, @(x) imresize(x, targetSize));
imgSetTest = transform(imgSetTestRaw,@(x) imresize(x,targetSize));

% do the same for the segmentation sets
segSetTrainRaw = subset(pxds, randomIndexes(1:34));
segSetValidateRaw = subset(pxds, randomIndexes(35:40));
segSetTestRaw = subset(pxds, randomIndexes(41:50));
segSetTrain = transform(segSetTrainRaw, @(x) {imresize(x{1}, targetSize, 'nearest')});     
segSetValidate = transform(segSetValidateRaw, @(x) {imresize(x{1}, targetSize, 'nearest')});
segSetTest  = transform(segSetTestRaw,  @(x) {imresize(x{1}, targetSize, 'nearest')});

% a lot of inspiration taken from constructing the CNN as a 
% graph rather than a linear structure - allows us to implement
% unet style skipping layers and other topological benefits
numClasses = 3;
net = dlnetwork;
inputSize = [240 320 3];

encode_layers = [
    imageInputLayer(inputSize, Normalization="zscore")

    % each "layer" composed as a unet layer, 2 convs and matching BN
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
   convolution2dLayer(3, 128, 'Padding', 1, 'Name', 'bottleneck_1')
   batchNormalizationLayer("Name","batch_bn_1int")
   reluLayer("Name","relu_intbn") 
   convolution2dLayer(3, 128, 'Padding', 1, 'Name', 'bottleneck_2')
   batchNormalizationLayer("Name","batch_bn_1")
   reluLayer("Name","relu_bn")

   % dropout to reduce overfitting - heavier dropout here as augmentation
   dropoutLayer(0.4, "Name","dropout")
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
% for improved network, this is where the augmentation takes place ONLY on
% training data, not on validation data.
trainingData = transform(combine(imgSetTrain, segSetTrain), @augmentData);
validationData = combine(imgSetValidate, segSetValidate);

% training hyperparameters
opts = trainingOptions('adam', ...
   'InitialLearnRate',1e-3, ...
   'MaxEpochs',50,...
   'MiniBatchSize',4, ...
   'LearnRateSchedule','piecewise',...
   'LearnRateDropPeriod',10, ...
   'LearnRateDropFactor',0.5, ...
   'ValidationData',validationData,...
   'ValidationFrequency',4,...
   'ValidationPatience',30, ...
   'OutputNetwork', 'best-validation-loss',...
   'Plots','training-progress', ...
   'Metrics','accuracy'...
   );

% change this to true when you want to train a new model (est. 3-4 minutes)
trainNewModel = false;  

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

% open results directory for both imp and base.
outputDir = fullfile(pwd, 'segmentationImprovedResults');
baseDir = fullfile(pwd, 'segmentationResults');
if ~exist(outputDir, 'dir'); mkdir(outputDir); end

fullSize = [966 1296];
i = 1;
while hasdata(imgSetTest)
    img = read(imgSetTest);

    % instead of just resizing images back to original resolution, we can
    % exploit bilinear sampling to upsample them back 4x bigger than
    % before.

    % improved - get the scores then bilinearly interpolate to resize
    [~, ~, scoresImp] = semanticseg(img, netImp);
    scoresUp = imresize(scoresImp, fullSize);          
    [~, predIdx] = max(scoresUp, [], 3);
    predFull = categorical(double(predIdx), 1:numClasses, cellstr(classNames));
    imwrite(label2rgb(uint8(predFull), [0 0 0; 1 0 0; 0 1 0]), fullfile(outputDir, sprintf('prediction_%02d.png', i)));

    % base - same as for improved.
    [~, ~, scoresBase] = semanticseg(img, netBase);
    scoresUp = imresize(scoresBase, fullSize);
    [~, predIdx] = max(scoresUp, [], 3);
    predFull = categorical(double(predIdx), 1:numClasses, cellstr(classNames));
    imwrite(label2rgb(uint8(predFull), [0 0 0; 1 0 0; 0 1 0]), fullfile(baseDir, sprintf('prediction_%02d.png', i)));

    i = i + 1;
end

% read results back in after they've been generated
pxdsResults = pixelLabelDatastore(outputDir, classNames, {[0 0 0], [255 0 0], [0 255 0]});
pxdsBaseResults = pixelLabelDatastore(baseDir, classNames, {[0 0 0], [255 0 0], [0 255 0]});

% evaluate the segmentation against test sets
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

% display conf matrixes for both
figure;
tiledlayout(1,2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
cm = confusionchart(metrics.ConfusionMatrix.Variables, classNames, Normalization="row-normalized");
cm.Title = "Normalised Confusion Matrix";
nexttile;
cm2 = confusionchart(metrics2.ConfusionMatrix.Variables, classNames, Normalization="row-normalized");
cm2.Title = "Normalised Base Confusion Matrix";

% display a set of results to see how well we did.
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

% see loss function ref - in improved upgraded to FOCAL rather than CE, AND
% used median frequency for imbalance 
function customLossFunction = diceAndFocal(predictions, truths, frequencies)
    % frequencies - used to calculate weighting
    % predictions - our normalised predictions
    % truths - the actual classification of the pixels

    % calculate the weights
    classWeights = median(frequencies) ./ frequencies;
    classWeights = reshape(classWeights, 1, 1, []);

    % calculate dice loss
    dice = 1 - mean(generalizedDice(predictions, truths), "all");
    
    % calclate focal loss
    gamma = 2;
    pt = sum(predictions .* truths, 3);  
    alpha_t = sum(classWeights .* truths, 3);
    focal = -mean(alpha_t .* (1 - pt).^gamma .* log(pt + 1e-8), "all");

    % add them together with heavier weighting on dice
    customLossFunction = 0.7*dice + 0.3*focal;
end

% augmentation function
function output = augmentData(images)
    
    image = images{1};
    segmentation = images{2};
    
    % geometric transformations
    % left right flip
    if rand < 0.5; image = fliplr(image); segmentation = fliplr(segmentation); end

    % up down flip
    if rand < 0.5; image = flipud(image); segmentation = flipud(segmentation); end
    

    % photometric transformations
    img = rgb2hsv(im2single(image)); % image has to be in HSV rather than RGB
    
    % jitters: a lot of the images base on colour, so photometric
    % approaches are strong here.

    % jitter hue with a prob of 0.6
    if rand < 0.6
      img(:,:,1) = mod(img(:,:,1) + (rand * 0.06 - 0.03), 1.0);
    end
    
    % jitter saturation with prob of 0.5
    if rand < 0.5
      img(:,:,2) = img(:,:,2) * (0.7 + rand * 0.6);  % saturation jitter
      img(:,:,2) = min(img(:,:,2), 1.0);
    end

    % convert back to RGB
    img = hsv2rgb(img);

    % gamma correction in range 0.7 to 1.3
    if rand < 0.4
        gamma = 0.7 + rand * 0.6;  
        img = img .^ gamma;
    end

    if rand < 0.3
      img = imnoise(img, 'gaussian', 0, 0.002);
    end

    % convert back to img
    img = im2uint8(min(max(img, 0), 1));
    image = img;

    output = {image, segmentation};
end