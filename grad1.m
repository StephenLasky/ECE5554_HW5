% this is the code for the first graduate question
% reference, EM alg is soft?: https://arxiv.org/pdf/1302.1552.pdf
% src for hard vs. soft: https://www.youtube.com/watch?v=ThgJzGYVWzc

% for this graduate question we simply copy the code from part 3 and modify
% it such that it uses soft clustering instead of hard clustering

% numTrainIms = 1888;
% numTestIms = 800;

numTrainIms = 25;
numTestIms = 50;
k = 5;
siftDimensions = 128;
numKMiter = 10;
numCategories = 8;

% rng(2); % seed random for predictable results

ims = generateImageCells(numTrainIms, 'train/');        % to generate
load('sift_desc.mat');
load('gs.mat');

% initialize clusters randomly
% access using train_D(i){1,1}

% first compute the possible ranges of the clusters
sampleSize = min(numTrainIms,10);
descriptorRanges = zeros(siftDimensions, 2, 'uint8');   % 2: one for min, one for max
descriptorRanges(:,1) = realmax;
for im_num = 1:sampleSize
    descriptor = train_D(im_num);
    descriptor = descriptor{1,1};
    dMin = min(descriptor')';        % 1x128
    dMax = max(descriptor')';        % 1x128
    descriptorRanges(:,1) = min(dMin, descriptorRanges(:,1));
    descriptorRanges(:,2) = max(dMax, descriptorRanges(:,2));
end

% next begin generating the random clusters
clusters = zeros(siftDimensions, k, 'single');
for dim = 1:siftDimensions
    clusters(dim,:) = randi(descriptorRanges(dim,2),k,1);
end
maxClusterInitValue = max(max(clusters));
% clusters

% compute the number of descriptors for each image
numDescriptors = zeros(numTrainIms,1);
for im_num = 1:numTrainIms
    descriptor = train_D(im_num);
    descriptor = descriptor{1,1};
    s = size(descriptor);
    s = s(2);
    numDescriptors(im_num) = s;
end

% for each and every descriptor, compute the "cluster belongingness" to
% every cluster based on the distance
totalNumDescriptors = sum(numDescriptors);
d2cBelongingness = zeros(k, totalNumDescriptors, 'single');
start_idx = 1;
end_idx = 1;
trainX = zeros(k,numTrainIms,'single');
for im_num = 1:numTrainIms
    descriptor = train_D(im_num);
    descriptor = single(descriptor{1,1});
    num_d = numDescriptors(im_num);
    end_idx = start_idx + num_d -1;
    
    % compute distances to every cluster
    distances = pdist2(clusters', descriptor');
    d2cBelongingness(:,start_idx:end_idx) = distances;
    
    % now update the scores such that they are a ratio of distance / total
    % distance. This will give a % of how much each should be.
    for d = start_idx:end_idx
        d2cBelongingness(:,d) = d2cBelongingness(:,d) / sum(d2cBelongingness(:,d));
        trainX(:,im_num) = trainX(:,im_num) + d2cBelongingness(:,d);
    end
    
    % update the start index
    start_idx = start_idx + num_d; 
end








