%% Load dbn data

dbn = load('../data/mnist/dbn-500-500-2000/dbn-500-500-2000.mat')';

%% Sample a digit just to check

digit = 5;

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
    vis(end-9:end) = indicator;
    %find(vis(end-9:end)) - 1
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

