rootFolder = fullfile('C:\Users\vaitk\Desktop\IntelektikosProjektas', 'Raides');

categories = {'à', 'è', 'æ'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

tbl = countEachLabel(imds)

minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)

[trainingSet, validationSet] = splitEachLabel(imds, 0.3, 'randomize');

% Find the first instance of an image for each category
aNosine = find(trainingSet.Labels == 'à', 1);
cSuVarnele = find(trainingSet.Labels == 'è', 1);
eNosine = find(trainingSet.Labels == 'æ', 1);

% figure

subplot(1,3,1);
imshow(readimage(trainingSet,aNosine))
subplot(1,3,2);
imshow(readimage(trainingSet,cSuVarnele))
subplot(1,3,3);
imshow(readimage(trainingSet,eNosine))

bag = bagOfFeatures(trainingSet);

img = readimage(imds, 1);
featureVector = encode(bag, img);

% Plot the histogram of visual word occurrences
figure
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')

categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);

confMatrix = evaluate(categoryClassifier, trainingSet);

confMatrix = evaluate(categoryClassifier, validationSet);

% Compute average accuracy
mean(diag(confMatrix));

img = imread(fullfile(rootFolder, 'è', '1.jpg'));
[labelIdx, scores] = predict(categoryClassifier, img);

% Display the string label
categoryClassifier.Labels(labelIdx)