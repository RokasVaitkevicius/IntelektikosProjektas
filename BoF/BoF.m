close all;
%rootFolder = fullfile('C:\Users\vaitk\Desktop\IntelektikosProjektas', 'Raides');
rootFolder = fullfile('C:\Users\vaitk\Desktop\IntelektikosProjektas\BoF', 'Raides');
testFolder = fullfile('C:\Users\vaitk\Desktop\IntelektikosProjektas\BoF', 'TestuojamosRaides');
categories = {'�', '�', '�', '�', '�', '�', '�', '�', '�'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imdts = imageDatastore(fullfile(testFolder, categories), 'LabelSource', 'foldernames');

trainingSet = imds;
validationSet = imdts;
%[trainingSet, validationSet] = splitEachLabel(imds, 0.5);
tsc = countEachLabel(trainingSet)
vsc = countEachLabel(validationSet)

% Tiesiog paziureti ar geroje vietoje iesko raidziu
aNosine = find(validationSet.Labels == '�', 1);
cSuVarnele = find(validationSet.Labels == '�', 1);
eNosine = find(validationSet.Labels == '�', 1);
subplot(1,3,1);
imshow(readimage(validationSet,aNosine))
subplot(1,3,2);
imshow(readimage(validationSet,cSuVarnele))
subplot(1,3,3);
imshow(readimage(validationSet,eNosine))


% Apmokome algoritma
bag = bagOfFeatures(trainingSet);

img = readimage(imds, 1);
featureVector = encode(bag, img);
categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);

% Visu pirma isbandome savo apmokyta algoritma su testavimo setu,
% kad paziuretume jog gerai veiki. Tikslumas turi buti pakankamai didelis
confMatrix = evaluate(categoryClassifier, trainingSet);

% Testuojame su naujais duomenimis ir ziurime kokias reiksmes gausim
confMatrix = evaluate(categoryClassifier, validationSet);

% Vidutinis tikslumas
mean(diag(confMatrix));

img = imread(fullfile(rootFolder, '�', '1.jpg'));
[labelIdx, scores] = predict(categoryClassifier, img);

% Spejama raide
categoryClassifier.Labels(labelIdx)