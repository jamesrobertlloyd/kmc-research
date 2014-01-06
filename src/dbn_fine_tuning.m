%% Clear

clear all;
close all;
addpath(genpath('./util'));

%% Load dbn data

dbn = load('../data/mnist/dbn-500-500-2000/dbn-500-500-2000.mat')';

%% Sample a digit just to check

digit = 2;

indicator = zeros(10,1);
indicator(digit+1) = 1;
vis = [(rand(500, 1) > 0.5) * 1; indicator];

% GS in top level

for iter = 1:500
    pre_sig = dbn.W2' * vis + dbn.h2;
    hid_prob = 1 ./ (1 + exp(-pre_sig));
    hid = (hid_prob > rand(size(hid_prob))) * 1;
    pre_sig = dbn.W2 * hid + dbn.v2;
    vis_prob = 1 ./ (1 + exp(-pre_sig));
    vis = (vis_prob > rand(size(vis_prob))) * 1;
    % Clamp
    %vis(end-9:end) = indicator;
    find(vis(end-9:end)) - 1
    % Propogate down
    pre_sig = dbn.W1 * vis(1:500) + dbn.v1;
    vis_prob = 1 ./ (1 + exp(-pre_sig));
    vis1 = (vis_prob > rand(size(vis_prob))) * 1;

    pre_sig = dbn.W0 * vis1 + dbn.v0;
    vis_prob = 1 ./ (1 + exp(-pre_sig));

    % Display

    imagesc(-reshape(vis_prob, 28, 28)');
    colormap('bone');
    drawnow;
end

%% Load mnist training data

load '../data/mnist/mnist_all.mat'
train_digits = [train0;
               train1;
               train2;
               train3;
               train4;
               train5;
               train6;
               train7;
               train8;
               train9];
train_labels = 0 * ones(size(train0,1),1);
train_labels = [train_labels; 1 * ones(size(train1,1),1)];
train_labels = [train_labels; 2 * ones(size(train2,1),1)];
train_labels = [train_labels; 3 * ones(size(train3,1),1)];
train_labels = [train_labels; 4 * ones(size(train4,1),1)];
train_labels = [train_labels; 5 * ones(size(train5,1),1)];
train_labels = [train_labels; 6 * ones(size(train6,1),1)];
train_labels = [train_labels; 7 * ones(size(train7,1),1)];
train_labels = [train_labels; 8 * ones(size(train8,1),1)];
train_labels = [train_labels; 9 * ones(size(train9,1),1)];

%% Standardise digit data

train_digits = double(train_digits);
train_digits = train_digits / max(max(train_digits));

%% Randomize order of mnist

perm = randperm(size(train_digits, 1));
train_digits = train_digits(perm,:);
train_labels = train_labels(perm,:);

%% Setup for fine tuning

hidrecbiases = dbn.h0';
penrecbiases = dbn.h1';
topbiases = dbn.h2';

vishid = dbn.W0;
hidvis = dbn.W0';
hidpen = dbn.W1;
penhid = dbn.W1';
pentop = dbn.W2(1:500,:);
labtop = dbn.W2(501:510,:);

pengenbiases = dbn.v2(1:500)';
labgenbiases = dbn.v2(501:510)';
hidgenbiases = dbn.v1';
visgenbiases = dbn.v0';

r = 0.002;
epochs = 15;
numCDiters = 15;

%% Fine tuning

starting_epoch = 6;

for epoch = starting_epoch:epochs
    
    perm = randperm(size(train_digits, 1));
    train_digits = train_digits(perm,:);
    train_labels = train_labels(perm,:);

    for i = 1:size(train_digits, 1)

        fprintf('epoch %d of %d iter %d of %d\n', epoch, epochs, i, ...
                size(train_digits, 1));
        
        data = train_digits(i,:);
        targets = zeros(1,10);
        targets(train_labels(i) + 1) = 1;

        % UP-DOWN ALGORITHM
        %
        % the data and all biases are row vectors.
        % the generative model is: lab <--> top <--> pen --> hid --> vis
        % the number of units in layer foo is numfoo
        % weight matrices have names fromlayer_tolayer
        % "rec" is for recognition biases and "gen" is for generative biases.
        % for simplicity, the same learning rate, r, is used everywhere.
        % PERFORM A BOTTOM-UP PASS TO GET WAKE/POSITIVE PHASE PROBABILITIES
        % AND SAMPLE STATES
        wakehidprobs = logistic(data*vishid + hidrecbiases);
        wakehidstates = (wakehidprobs > rand(size(wakehidprobs))) * 1;
        poshidstates = wakehidstates;
        wakepenprobs = logistic(wakehidstates*hidpen + penrecbiases);
        wakepenstates = (wakepenprobs > rand(size(wakepenprobs))) * 1;
        %postopprobs = logistic(wakepenstates*pentop + targets*labtop + topbiases);
        %postopstates = waketopprobs > rand(size(waketopprobs));
        waketopprobs = logistic(wakepenstates*pentop + targets*labtop + topbiases);
        waketopstates = (waketopprobs > rand(size(waketopprobs))) * 1;
        % 
        % POSITIVE PHASE STATISTICS FOR CONTRASTIVE DIVERGENCE
        poslabtopstatistics = targets' * waketopstates;
        pospentopstatistics = wakepenstates' * waketopstates;
        % PERFORM numCDiters GIBBS SAMPLING ITERATIONS USING THE TOP LEVEL
        % UNDIRECTED ASSOCIATIVE MEMORY
        negtopstates = waketopstates; % to initialize loop
        for iter=1:numCDiters
            negpenprobs = logistic(negtopstates*pentop' + pengenbiases);
            negpenstates = (negpenprobs > rand(size(negpenprobs))) * 1;
            neglabprobs = softmax(negtopstates*labtop' + labgenbiases);
            %neglabprobs = logistic(negtopstates*labtop' + labgenbiases);
            negtopprobs = logistic(negpenstates*pentop+neglabprobs*labtop+topbiases);
            negtopstates = (negtopprobs > rand(size(negtopprobs))) * 1;
        end
        % 
        % NEGATIVE PHASE STATISTICS FOR CONTRASTIVE DIVERGENCE
        negpentopstatistics = negpenstates'*negtopstates;
        neglabtopstatistics = neglabprobs'*negtopstates;
        % STARTING FROM THE END OF THE GIBBS SAMPLING RUN, PERFORM A
        % TOP-DOWN GENERATIVE PASS TO GET SLEEP/NEGATIVE PHASE PROBABILITIES
        % AND SAMPLE STATES
        sleeppenstates = negpenstates;
        sleephidprobs = logistic(sleeppenstates*penhid + hidgenbiases);
        sleephidstates = (sleephidprobs > rand(size(sleephidprobs))) * 1;
        sleepvisprobs = logistic(sleephidstates*hidvis + visgenbiases);
        % PREDICTIONS
        psleeppenstates = logistic(sleephidstates*hidpen + penrecbiases);
        psleephidstates = logistic(sleepvisprobs*vishid + hidrecbiases);
        pvisprobs = logistic(wakehidstates*hidvis + visgenbiases);
        phidprobs = logistic(wakepenstates*penhid + hidgenbiases);
        % UPDATES TO GENERATIVE PARAMETERS
        hidvis = hidvis + r*poshidstates'*(data-pvisprobs);
        visgenbiases = visgenbiases + r*(data - pvisprobs);
        penhid = penhid + r*wakepenstates'*(wakehidstates-phidprobs);
        hidgenbiases = hidgenbiases + r*(wakehidstates - phidprobs);
        % UPDATES TO TOP LEVEL ASSOCIATIVE MEMORY PARAMETERS
        labtop = labtop + r*(poslabtopstatistics-neglabtopstatistics);
        labgenbiases = labgenbiases + r*(targets - neglabprobs);
        pentop = pentop + r*(pospentopstatistics - negpentopstatistics);
        pengenbiases = pengenbiases + r*(wakepenstates - negpenstates);
        topbiases = topbiases + r*(waketopstates - negtopstates);
        %UPDATES TO RECOGNITION/INFERENCE APPROXIMATION PARAMETERS
        hidpen = hidpen + r*(sleephidstates'*(sleeppenstates-psleeppenstates));
        penrecbiases = penrecbiases + r*(sleeppenstates-psleeppenstates);
        vishid = vishid + r*(sleepvisprobs'*(sleephidstates-psleephidstates));
        hidrecbiases = hidrecbiases + r*(sleephidstates-psleephidstates);
    
    end
        
end

%% Reconstruct into dbn

dbn_ft = struct('W2', [pentop; labtop], 'W1', penhid', 'W0', hidvis', ...
                'v2', [pengenbiases'; labgenbiases'], ...
                'v1', hidgenbiases', 'v0', visgenbiases', ...
                'h2', topbiases');
            
%% Sample a digit from the fine tuned dbn

digit = 0;

indicator = zeros(10,1);
indicator(digit+1) = 1;
vis = [(rand(500, 1) > 0.5) * 1; indicator];

% GS in top level

for iter = 1:500
    pre_sig = dbn_ft.W2' * vis + dbn_ft.h2;
    hid_prob = 1 ./ (1 + exp(-pre_sig));
    hid = (hid_prob > rand(size(hid_prob))) * 1;
    pre_sig = dbn_ft.W2 * hid + dbn_ft.v2;
    vis_prob = 1 ./ (1 + exp(-pre_sig));
    vis = (vis_prob > rand(size(vis_prob))) * 1;
    % Clamp
    %vis(end-9:end) = indicator;
    find(vis(end-9:end)) - 1
    % Propogate down
    pre_sig = dbn_ft.W1 * vis(1:500) + dbn_ft.v1;
    vis_prob = 1 ./ (1 + exp(-pre_sig));
    vis1 = (vis_prob > rand(size(vis_prob))) * 1;

    pre_sig = dbn_ft.W0 * vis1 + dbn_ft.v0;
    vis_prob = 1 ./ (1 + exp(-pre_sig));

    % Display

    imagesc(-reshape(vis_prob, 28, 28)');
    colormap('bone');
    drawnow;
end