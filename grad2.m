% numTrainIms = 1888;
% numTestIms = 800;

run('vlfeat-0.9.20/toolbox/vl_setup')

numTrainIms = 100;
numTestIms = 50;
k = 25;
siftDimensions = 128;
numKMiter = 10;
numCategories = 8;

% rng(2); % seed random for predictable results

ims = generateImageCells(numTrainIms, 'train/');        % to generate
load('sift_desc.mat');
load('gs.mat');

% compute the number of descriptors for each image
numDescriptors = zeros(numTrainIms,1);
for im_num = 1:numTrainIms
    descriptor = train_D(im_num);
    descriptor = descriptor{1,1};
    s = size(descriptor);
    s = s(2);
    numDescriptors(im_num) = s;
end

% GRAD QUESTION 2 CODE HERE: REPLACES STANDARD CLUSTER GENERATION
% first put all descriptors into one massive matrix
totalNumDescriptors = sum(numDescriptors);
descriptors = zeros(siftDimensions, totalNumDescriptors, 'single');
start_idx = 1;
end_idx = 1;
for im_num = 1:numTrainIms
    descriptor = train_D(im_num);
    descriptor = single(descriptor{1,1});
    num_d = numDescriptors(im_num);
    end_idx = start_idx + num_d -1;
    
    % add to descriptors matrix
    descriptors(:,start_idx:end_idx) = descriptor;
    
    % update the start index
    start_idx = start_idx + num_d; 
end



[means, covariances, priors] = vl_gmm(descriptors, k);


% GRAD QUESTION 2 END. SEE ABOVE.

% % now generate trainX
% % trainX is k x numTrainIms
% trainX = zeros(k,numTrainIms, 'single');
% for im_num = 1:numTrainIms
%     for d = 1:numDescriptors(im_num)
%         c = dictionary(im_num, d);
%         trainX(c,im_num) = trainX(c,im_num) + 1;
%     end
% end
% 
% % now generate trainY
% trainY = train_gs(1:numTrainIms);
% 
% % BEGIN: classify test images
% % start by filling in testX
% testX = zeros(k,numTestIms,'single');
% testY = zeros(1,numTestIms);
% for im_num = 1:numTestIms
%     
%     % get the set of descriptors for this image
%     descriptor = test_D(im_num);
%     descriptor = single(descriptor{1,1});
%     num_d = size(descriptor); num_d = num_d(2);
%     
%     % compute distances and min indexes
%     distances = pdist2(clusters', descriptor');
%     [vals,clust_idxs] = min(distances); 
%     
%     % fill in testX
%     for clust_idx = 1:num_d
%         testX(clust_idxs(clust_idx), im_num) = testX(clust_idxs(clust_idx), im_num) + 1;
%     end   
% end
% 
% % we need to train on each of the difference categories
% imClasScores = zeros(numCategories, numTestIms, 'single');
% for cat = 1:numCategories
%     selection = trainY == cat;
%     selection = int8(selection);
%     svm = fitcsvm(trainX', selection');
%     [label, score] = predict(svm, testX');
%     
%     imClassScores(cat,:) = score(:,2)';
% end
% 
% % now that each test image has a score, it is time to figure out which
% % image goes where
% [vals, maxScoreCat] = max(imClassScores);
% testY = maxScoreCat;
% 
% % END: classify test images
% 
% % test acuracy
% actualTestY = test_gs(1:numTestIms);
% accuracy = testY == actualTestY;
% accuracy = single(sum(accuracy) / numTestIms)    