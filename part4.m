% numTrainIms = 1888;
% numTestIms = 800;

numTrainIms = 250;
numTestIms = 50;
k = 50;
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

% now implement the k-means algorithm
dictionary = zeros(numTrainIms, max(numDescriptors));
clusterAssignments = zeros(k,1);    % tracks the number of assignments to each cluster
clusterSum = clusters;     % tracks the total value of the cluster for mean
clusterSumUpdated = false(k,1); % tracks whether or not the cluster sum has been updated

% start iterating ...
for iter = 1:numKMiter
    for im_num = 1:numTrainIms
        % step 1: for each x: assign to closest cluster c
        % get descriptor
        descriptor = train_D(im_num);
        descriptor = single(descriptor{1,1});

        % compute distances and min indexes
        distances = pdist2(clusters', descriptor');
        [vals,idxs] = min(distances); 

        % assign to dictionary
        s = numDescriptors(im_num);
        dictionary(im_num,1:s) = idxs;

        % step 2: recompute each cluster center as average of assignments
        % (continued)
        for d = 1:s
            c = dictionary(im_num, d);
            % if the cluster has not been updated yet, set everything to
            % zero. we do this such that unused clusters do not get set to
            % zero for some reason. 
            if clusterSumUpdated(c) == false
                clusterSumUpdated(c) = true;
                clusterSum(:,c) = 0;
            end
            
            clusterSum(:,c) = clusterSum(:,c) + descriptor(:,d);
            clusterAssignments(c) = clusterAssignments(c) + 1;
        end
    end
    
    % baby the clusterAssignments to ensure we do not start losing clusters.
    % (division by zero)
    clusterAssignments = max(clusterAssignments,1);
    
    % finally update the cluster values
    clusters = single(clusterSum ./ repmat(clusterAssignments,1,128)');

%     % destory vacant clusters
%     clusters = clusters(:,all(~isnan(clusters)));    % for nan - columns
%     newK = size(clusters); newK = newK(2); k = newK; % update k

    % re-initialize unused clusters
    for c = 1:k
        if ~clusterSumUpdated(c)
            clusters(:,c) = randi(maxClusterInitValue,siftDimensions,1);
        end
    end

    % reset variables
    clusterSum(:) = 0;
    clusterAssignments(:) = 0;
    clusterSumUpdated(:) = false; 
end

% now generate trainX
% trainX is k x numTrainIms
trainX = zeros(5*k,numTrainIms, 'single');  % multiply by 5 now because we quadrants + whole
for im_num = 1:numTrainIms
    % compute img size for quadrants (later on)
    [sizeX, sizeY, sizeZ] = size(ims{im_num});
    
    % get the position of the relevant descriptors (later on)
    descriptorInfo = train_F(im_num);
    descriptorInfo = single(descriptorInfo{1,1});
    
    for d = 1:numDescriptors(im_num)
        c = dictionary(im_num, d);
        trainX(c,im_num) = trainX(c,im_num) + 1;
        
        % now do quadrants
        % for train_F: first element is X, second element is Y
        % we know which cluster this element belongs to
        % for each quadrant: (1) determine if belongs in (2) add
        x = descriptorInfo(1,d);
        y = descriptorInfo(2,d);
        
        % determine the quadrants
        if x <= sizeX / 2 && y <= sizeY / 2
            quadrant = 1;   % top left
        elseif x > sizeX / 2 && y <= sizeY / 2
            quadrant = 2;   % top right
        elseif x <= sizeX / 2 && y > sizeY / 2
            quadrant = 3;   % bottom left
        else
            quadrant = 4;    % bottom right
        end
        
        % this will be the standard offset we use for placement into trainX
        % as well as testX
        offset = k * quadrant;
        trainX(c + offset, im_num) = trainX(c + offset, im_num) + 1;
    end
end

% now generate trainY
trainY = train_gs(1:numTrainIms);

% BEGIN: classify test images
% get the testing images
ims = generateImageCells(numTestIms, 'test/');        % to generate

% start by filling in testX
testX = zeros(5*k,numTestIms,'single'); % multiply by 5 now because we quadrants + whole
testY = zeros(1,numTestIms);
for im_num = 1:numTestIms
    % compute img size for quadrants (later on)
    [sizeX, sizeY, sizeZ] = size(ims{im_num});
    
    % get the position of the relevant descriptors (later on)
    descriptorInfo = test_F(im_num);
    descriptorInfo = single(descriptorInfo{1,1});
    
    
    % get the set of descriptors for this image
    descriptor = test_D(im_num);
    descriptor = single(descriptor{1,1});
    num_d = size(descriptor); num_d = num_d(2);
    
    % compute distances and min indexes
    distances = pdist2(clusters', descriptor');
    [vals,clust_idxs] = min(distances); 
    
    % fill in testX
    quadrant = 0;
    for d = 1:num_d
        clust_idx = clust_idxs(d);
        testX(clust_idx, im_num) = testX(clust_idx, im_num) + 1;
        
        % now do quadrants
        % for train_F: first element is X, second element is Y
        % we know which cluster this element belongs to
        % for each quadrant: (1) determine if belongs in (2) add
        x = descriptorInfo(1,d);
        y = descriptorInfo(2,d);
        
        % determine the quadrants
        if x <= sizeX / 2 && y <= sizeY / 2
            quadrant = 1;   % top left
        elseif x > sizeX / 2 && y <= sizeY / 2
            quadrant = 2;   % top right
        elseif x <= sizeX / 2 && y > sizeY / 2
            quadrant = 3;   % bottom left
        else
            quadrant = 4;    % bottom right
        end
        
        % this will be the standard offset we use for placement into trainX
        % as well as testX
        offset = k * quadrant;
        testX(clust_idx + offset, im_num) = testX(clust_idx + offset, im_num) + 1;
    end
end

% we need to train on each of the difference categories
imClasScores = zeros(numCategories, numTestIms, 'single');
for cat = 1:numCategories
    selection = trainY == cat;
    selection = int8(selection);
    svm = fitcsvm(trainX', selection');
    [label, score] = predict(svm, testX');
    
    imClassScores(cat,:) = score(:,2)';
end

% now that each test image has a score, it is time to figure out which
% image goes where
[vals, maxScoreCat] = max(imClassScores);
testY = maxScoreCat;

% END: classify test images

% test acuracy
actualTestY = test_gs(1:numTestIms);
accuracy = testY == actualTestY;
accuracy = single(sum(accuracy) / numTestIms)    