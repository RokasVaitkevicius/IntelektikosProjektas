

close all;

% Load training and test data using |imageDatastore|.
syntheticDir   = fullfile('C:\Users\Dovydas\Desktop\kurs','testËøû');
handwrittenDir = fullfile('C:\Users\Dovydas\Desktop\kurs','duom3');

% |imageDatastore| recursively scans the directory tree containing the
% images. Folder names are automatically used as labels for each image.
trainingSet = imageDatastore(handwrittenDir,   'IncludeSubfolders', true, 'LabelSource', 'foldernames');
testSet     = imageDatastore(syntheticDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
 

countEachLabel(trainingSet)
countEachLabel(testSet)


figure;

subplot(2,3,1);
imshow(trainingSet.Files{13});

subplot(2,3,2);
imshow(trainingSet.Files{16});

subplot(2,3,3);
imshow(trainingSet.Files{21});

subplot(2,3,4);
imshow(testSet.Files{9});

subplot(2,3,5);
imshow(testSet.Files{13});

subplot(2,3,6);
imshow(testSet.Files{22});


% Show pre-processing results
exTestImage = readimage(testSet,10);
processedImage = imbinarize(rgb2gray(exTestImage));

figure;

subplot(1,2,1)
imshow(exTestImage)

subplot(1,2,2)
imshow(processedImage)

img = readimage(trainingSet, 101);
processedImage = imbinarize(rgb2gray(img));

% Extract HOG features and HOG visualization
[hog_2x2, vis2x2] = extractHOGFeatures(processedImage,'CellSize',[2 2]);
[hog_4x4, vis4x4] = extractHOGFeatures(processedImage,'CellSize',[4 4]);
[hog_8x8, vis8x8] = extractHOGFeatures(processedImage,'CellSize',[8 8]);

% Show the original image
figure; 
subplot(2,3,1:3); imshow(processedImage);

% Visualize the HOG features
subplot(2,3,4);  
plot(vis2x2); 
title({'CellSize = [2 2]'; ['Feature length = ' num2str(length(hog_2x2))]});

subplot(2,3,5);
plot(vis4x4); 
title({'CellSize = [4 4]'; ['Feature length = ' num2str(length(hog_4x4))]});

subplot(2,3,6);
plot(vis8x8); 
title({'CellSize = [8 8]'; ['Feature length = ' num2str(length(hog_8x8))]});

cellSize = [2 2];
hogFeatureSize = length(hog_2x2);

numImages = numel(trainingSet.Files);
trainingFeatures = zeros(numImages, hogFeatureSize, 'single');

for i = 1:numImages
    img = readimage(trainingSet, i);
    
    img = rgb2gray(img);
    
    % Apply pre-processing steps
    img = imbinarize(img);
    
    trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);  
end

% Get labels for each image.
trainingLabels = trainingSet.Labels;

classifier = fitcecoc(trainingFeatures, trainingLabels);

[testFeatures, testLabels] = helperExtractHOGFeaturesFromImageSet(testSet, hogFeatureSize, cellSize);

% Make class predictions using the test features.
predictedLabels = predict(classifier, testFeatures);

% Tabulate the results using a confusion matrix.
[confMat,order] = confusionmat(testLabels, predictedLabels)
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

digits = order;
colHeadings = arrayfun(@(x)sprintf('%s',x),     order,'UniformOutput',false);
format = repmat('  %-6s',1,11);
header = sprintf(format,'letters  |',colHeadings{:});
fprintf('\n%s\n%s\n',header,repmat('-',size(header)));
for idx = 1:numel(digits)
    fprintf('%-6s',   [digits(idx) '      |']);
    fprintf('%-9.2f', confMat(idx,:));
    fprintf('\n')
end

displayEndOfDemoMessage(mfilename)