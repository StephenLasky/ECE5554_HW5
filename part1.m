numTrainIms = 1888;
numTestIms = 800;
ims = generateImageCells(numTrainIms, 'train/');        % to generate
nBins = 12;
nCategories = 8;
load('gs.mat');
% figure(); imshow(ims{2});   % to show

% im = ims{1};
% s = size(im);
% imv = reshape(im, s(1)*s(2), 3);    % reshape to vector
% % im = [1 2 5 8 4 1 8 6 4 2 6];

%%%  STEP 1: 'TRAINING' PART %%%
% compute the avareage of each cluster center
clusters = zeros(nBins, 3,nCategories,'single');  % 3 for RGB channels
catCount = zeros(nCategories,1);
for im_num = 1:numTrainIms
    category = train_gs(im_num);
    catCount(category) = catCount(category) + 1;
    
    im = ims{im_num};
    s = size(im);
    imv = reshape(im, s(1)*s(2), 3);
    h = hist(imv,nBins);
    clusters(:, :, category) = clusters(:, :, category) + h;
end

% divide to compute the averages
for cat = 1:nCategories
    clusters(:,:,cat) = clusters(:,:,cat) / catCount(cat);
end

% reshape clusters into 2D matrix
clusters = reshape(clusters, nBins * 3, nCategories);

%%% STEP 2: 'TESTING' PART %%%
% we now have the cluster centers, now we can begin the k means algorithm
ims = generateImageCells(numTestIms, 'test/');        % to generate
catResult = zeros(1,numTestIms);

% perform K-means
for im_num = 1:numTestIms
    im = ims{im_num};
    s = size(im);
    imv = reshape(im, s(1)*s(2), 3);
    h = hist(imv,nBins);
    h = reshape(h,nBins * 3, 1);
    h = repmat(h,1,8);
    h = single(h);
    
    % compute the closest cluster: compute distance, then find min
    d = clusters - h;
    d = d .^2;
    d = sum(d);
    d = d .^(0.5);
    [val, idx] = min(d);        
    
    % add to cluster, update the cluster mean and other header information
    n = catCount(idx) + 1; catCount(idx) = n;   % add one to this category
    h = squeeze(h(:,1));
    clusters(:,idx) = ((n-1)/n) * clusters(:,idx) + h / n;  % update the cluser 
    
    % finally, record the result
    catResult(im_num) = idx;
end

% compute the error 
numCorrect = test_gs(1:numTestIms) == catResult(1,:);
numCorrect = single(numCorrect);
numCorrect = sum(numCorrect)
correctRate = numCorrect / numTestIms