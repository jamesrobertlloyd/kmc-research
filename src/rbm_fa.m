%% Clear and load mmd tools

clear all;
addpath(genpath('./mmd'));
addpath(genpath('./util'));

%% Load mnist images

load '../data/mnist/mnist_all.mat'
test_digits = [test0;
               test1;
               test2;
               test3;
               test4;
               test5;
               test6;
               test7;
               test8;
               test9];
test_labels = 0 * ones(size(test0,1),1);
test_labels = [test_labels; 1 * ones(size(test1,1),1)];
test_labels = [test_labels; 2 * ones(size(test2,1),1)];
test_labels = [test_labels; 3 * ones(size(test3,1),1)];
test_labels = [test_labels; 4 * ones(size(test4,1),1)];
test_labels = [test_labels; 5 * ones(size(test5,1),1)];
test_labels = [test_labels; 6 * ones(size(test6,1),1)];
test_labels = [test_labels; 7 * ones(size(test7,1),1)];
test_labels = [test_labels; 8 * ones(size(test8,1),1)];
test_labels = [test_labels; 9 * ones(size(test9,1),1)];

%% Load rbm fantasies

fantasies = csvread('../data/mnist/rbm-samples/images.csv');
labels = csvread('../data/mnist/rbm-samples/labels.csv');
num_images = 3000;

%% Load multi rbm fantasies

fantasies = csvread('../data/mnist/many-rbm-samples/images.csv');
labels = csvread('../data/mnist/many-rbm-samples/labels.csv');
num_images = 1500;

%% Load dbn fantasies

fantasies = csvread('../data/mnist/dbn-samples/images.csv');
labels = csvread('../data/mnist/dbn-samples/labels.csv');
num_images = 3000;

%% Load dbn 500 500 2000 fantasies

fantasies = csvread('../data/mnist/dbn-500-500-2000/images.csv');
labels = csvread('../data/mnist/dbn-500-500-2000/labels.csv');
num_images = 3000;

%% Load dbn 500 500 2000 fine tuned fantasies

fantasies = csvread('../data/mnist/dbn-ft-samples/images.csv');
labels = csvread('../data/mnist/dbn-ft-samples/labels.csv');
num_images = 3000;
           
%% Standardise digit data

test_digits = double(test_digits);
test_digits = test_digits / max(max(test_digits));
fantasies = fantasies / max(max(fantasies));

%% Randomize order of test digits

perm = randperm(size(test_digits, 1));
test_digits = test_digits(perm,:);
test_labels = test_labels(perm,:);

%% Display random fantasies

i = randi(num_images);
imagesc(reshape(fantasies(i,:), 28, 28)');
display(labels(i));

%% Display many random fantasies

rows = 2;
cols = 15;
raster = [];
for row = 1:rows
    one_line = [];
    for col = 1:cols
        i = randi(num_images);
        one_line = [one_line, reshape(fantasies(i,:), 28, 28)'];
    end
    raster = [raster; one_line];
end
h = figure('Position', [300, 300, 000+size(raster,2), 000+size(raster,1)]);
imagesc(-raster);
colormap(bone);
set(gca, 'YTick', []);
set(gca, 'XTick', []);
save2pdf( 'samples.pdf', h, 600, true );

%% Extract digits
    
X = test_digits(1:num_images,:);
X_labels = test_labels(1:num_images);
Y = fantasies(1:num_images,:);
Y_labels = labels(1:num_images);

%% Extract digits - null hypothesis

X = test_digits(1:num_images,:);
X_labels = test_labels(1:num_images);
Y = test_digits(num_images:(num_images+num_images-1),:);
Y_labels = test_labels(num_images:(num_images+num_images-1));

%% No pre-processing

d = 784;
X_dr = X;
Y_dr = Y;

%% Perform PCA preprocessing

d = 5;
coeff = pca([X;Y]);
X_dr = X * coeff(:,1:d);
Y_dr = Y * coeff(:,1:d);

h = figure;
plot(X_dr(:,1), X_dr(:,2), 'go'); hold on;
plot(Y_dr(:,1), Y_dr(:,2), 'rx'); hold off;
save2pdf('temp/pca.pdf', h, 600, true );

%% Perform random projection preprocessing

d = 2;
coeff = randn(size(X,2));
X_dr = X_dr * coeff(:,1:d);
Y_dr = Y_dr * coeff(:,1:d);
plot(X_dr(:,1), X_dr(:,2), 'go');
hold on;
plot(Y_dr(:,1), Y_dr(:,2), 'rx');
hold off;

%% Perform factor analysis preprocessing

d = 2;

[L,psi,T,stats,F] = factoran([X;Y], 2);

X_dr = F(1:size(X,1),:);
Y_dr = F((size(X,1)+1):end,:);

plot(X_dr(:,1), X_dr(:,2), 'ro'); hold on;
plot(Y_dr(:,1), Y_dr(:,2), 'go'); hold off;

%% Calculate some distances for reference

d1 = sqrt(sq_dist(X_dr', X_dr'));
d2 = sqrt(sq_dist(Y_dr', Y_dr'));
Z_dr = [X_dr;X_dr];  %aggregate the sample
d3 = sq_dist(Z_dr', Z_dr');
hist([d1(:);d2(:)]);

%% Select a lengthscale

% CV for density estimation
folds = 5;
divisions = 50;
distances = sort([d1(:); d2(:)]);
%trial_ell = zeros(divisions-1,1);
trial_ell = zeros(divisions,1);
for i = 1:(divisions)%-1)
    %trial_ell(i) = distances(floor(i*numel(distances)/(2*divisions)));
    trial_ell(i) = i * sqrt(0.5) * distances(floor(0.5*numel(distances))) / divisions;
end
m = size(X_dr, 1);
n = size(Y_dr, 1);
d = size(X_dr, 2);
X_perm = X_dr(randperm(m),:);
Y_perm = Y_dr(randperm(n),:);
X_f_train = cell(folds,1);
X_f_test = cell(folds,1);
Y_f_train = cell(folds,1);
Y_f_test = cell(folds,1);
for fold = 1:folds
    if fold == 1
        X_f_train{fold} = X_perm(floor(fold*m/folds):end,:);
        X_f_test{fold} = X_perm(1:(floor(fold*m/folds)-1),:);
        Y_f_train{fold} = Y_perm(floor(fold*n/folds):end,:);
        Y_f_test{fold} = Y_perm(1:(floor(fold*n/folds)-1),:);
    elseif fold == folds
        X_f_train{fold} = X_perm(1:floor((fold-1)*m/folds),:);
        X_f_test{fold} = X_perm(floor((fold-1)*m/folds + 1):end,:);
        Y_f_train{fold} = Y_perm(1:floor((fold-1)*n/folds),:);
        Y_f_test{fold} = Y_perm(floor((fold-1)*m/folds + 1):end,:);
    else
        X_f_train{fold} = [X_perm(1:floor((fold-1)*m/folds),:);
                           X_perm(floor((fold)*m/folds+1):end,:)];
        X_f_test{fold} = X_perm(floor((fold-1)*m/folds + 1):floor((fold)*m/folds),:);
        Y_f_train{fold} = [Y_perm(1:floor((fold-1)*n/folds),:);
                           Y_perm(floor((fold)*n/folds+1):end,:)];
        Y_f_test{fold} = Y_perm(floor((fold-1)*n/folds + 1):floor((fold)*n/folds),:);
    end
end
best_ell = trial_ell(1);
best_log_p = -Inf;
for ell = trial_ell'
    display(ell);
    log_p = 0;
    for fold = 1:folds
        K1 = rbf_dot(X_f_train{fold} , X_f_test{fold}, ell);
        p_X = (sum(K1, 1)' / m) / (ell^d);
        log_p = log_p + sum(log(p_X));
        K2 = rbf_dot(Y_f_train{fold} , Y_f_test{fold}, ell);
        p_Y = (sum(K2, 1)' / n) / (ell^d);
        log_p = log_p + sum(log(p_Y));
    end
    if log_p > best_log_p
        best_log_p = log_p;
        best_ell = ell;
    end
end
params.sig = best_ell;
% Median
%params.sig = sqrt(0.5*median(d3(d3>0)));
% Other things?
%params.sig = 2;

display(params.sig);

%% Perform MMD test

alpha = 0.05;
params.shuff = 100;
[testStat,thresh,params,p] = mmdTestBoot_jl(X_dr,Y_dr,alpha,params);
display(p);

%% Compute witness function in 2d

if size(X_dr,2) == 2
    m = size(X_dr, 1);
    n = size(Y_dr, 1);
    t = (((fullfact([200,200])-0.5) / 200) - 0) * 1;
    t = t .* (1.4 * repmat(range([X_dr; Y_dr]), size(t,1), 1));
    t = t + repmat(min([X_dr; Y_dr]) - 0.2*range([X_dr; Y_dr]), size(t,1), 1);
    K1 = rbf_dot(X_dr, t, params.sig);
    K2 = rbf_dot(Y_dr, t, params.sig);
    witness = sum(K1, 1)' / m - sum(K2, 1)' / n;
    %plot3(t(:,1), t(:,2), witness, 'bo');
    %hold on;
    %plot3(Y_dr(:,1), Y_dr(:,2), repmat(max(max(witness)), size(Y_dr)), 'ro');
    reshaped = reshape(witness, 200, 200)';
    
    h = figure;
    imagesc(reshaped(end:-1:1,:));
    colorbar;
    save2pdf('temp/witness.pdf', h, 600, true );
    %hold off;
end

%% Compute witness function and show (least) favourite images

m = size(X_dr, 1);
n = size(Y_dr, 1);
t_X_dr = X_dr;
t_Y_dr = Y_dr;
K1 = rbf_dot(X_dr, t_X_dr, params.sig);
K2 = rbf_dot(Y_dr, t_X_dr, params.sig);
witness_X = sum(K1, 1)' / m - sum(K2, 1)' / n;
K1 = rbf_dot(X_dr, t_Y_dr, params.sig);
K2 = rbf_dot(Y_dr, t_Y_dr, params.sig);
witness_Y = sum(K1, 1)' / m - sum(K2, 1)' / n;

[~, i] = sort(witness_Y, 'ascend');

for j = 1:5
    imagesc(reshape(Y(i(j),:), 28, 28)');
    drawnow;
    pause;
end

[~, i] = sort(witness_X, 'descend');

for j = 1:5
    imagesc(reshape(X(i(j),:), 28, 28)');
    drawnow;
    pause;
end

%% Find peaks of the witness function on fantasies

close all;

ell = params.sig;
x_opt_Y = zeros(size(Y_dr));
witnesses_Y = zeros(size(Y_dr, 1), 1);

for i = 1:size(Y_dr,1)

    x = Y_dr(i,:)';
    witness = rbf_witness(x, X_dr, Y_dr, ell);
    witnesses_Y(i) = witness;
    fprintf('\nwitness=%f\n', witness);

    % Checkgrad
    % options = optimoptions('fminunc','GradObj','on', ...
    %                        'DerivativeCheck', 'on', ...
    %                        'FinDiffType', 'central');
    % fminunc(@(x) rbf_witness(x, X, Y, ell), x, options);  

    if witness >=0 
        % Maximize
        x = minimize_quiet(x, @(x) neg_rbf_witness(x, X_dr, Y_dr, ell), -50);
    else
        x = minimize_quiet(x, @(x) rbf_witness(x, X_dr, Y_dr, ell), -50);
    end
    x_opt_Y(i,:) = x';
    %imagesc(reshape(x, 28, 28)');
    %drawnow;
    
    fprintf('\ni = %d\n', i);
    
end

%% Find peaks of the witness function on test

close all;

ell = params.sig;
x_opt_X = zeros(size(X_dr));
witnesses_X = zeros(size(X_dr, 1), 1);

for i = 1:size(X_dr,1)

    x = X_dr(i,:)';
    witness = rbf_witness(x, X_dr, Y_dr, ell);
    witnesses_X(i) = witness;
    fprintf('\nwitness=%f\n', witness);

%     % Checkgrad
%     options = optimoptions('fminunc','GradObj','on', ...
%                            'DerivativeCheck', 'on', ...
%                            'FinDiffType', 'central');
%     fminunc(@(x) neg_rbf_witness(x, X, Y, ell), x, options);  

    if witness >=0 
        % Maximize
        x = minimize_quiet(x, @(x) neg_rbf_witness(x, X_dr, Y_dr, ell), -50);
    else
        x = minimize_quiet(x, @(x) rbf_witness(x, X_dr, Y_dr, ell), -50);
    end
    x_opt_X(i,:) = x';
    %imagesc(reshape(x, 28, 28)');
    %drawnow;
    
    fprintf('\ni = %d\n', i);
    
end

%% Use this to partition the space

d_YY = sq_dist(x_opt_Y', x_opt_Y');
d_YX = sq_dist(x_opt_Y', x_opt_X');
d_XX = sq_dist(x_opt_X', x_opt_X');
threshold = mean(d_YY(:)) / 100; % A better heuristic surely exists
c_Y = zeros(size(witnesses_Y));
c_X = zeros(size(witnesses_X));
c = 1;
for i = 1:length(c_Y)
    if c_Y(i) == 0
        c_Y(d_YY(i,:) < threshold) = c;
        c_X(d_YX(i,:) < threshold) = c;
        c = c + 1;
    end
end
for i = 1:length(c_X)
    if c_X(i) == 0
        c_X(d_XX(i,:) < threshold) = c;
        c = c + 1;
    end
end

max_c = c-1;
witness_sums = zeros(max_c,1);
witness_signs = zeros(max_c,1);
c_XY = [c_X; c_Y];
witnesses_XY = [witnesses_X; witnesses_Y];
for c = 1:max_c
    % TODO - these need to be weighted if m <> n
    witness_sums(c) = sum(witnesses_X(c_X==c)) - sum(witnesses_Y(c_Y==c));
    if sum(witnesses_XY(c_XY==c)) < 0
        witness_signs(c) = -1;
        %witness_sums(c) = -witness_sums(c);
    else
        witness_signs(c) = 1;
    end
end

MMD = sum(witness_sums);

if any(witness_sums < 0)
    warning('Uh oh!');
else
    witness_sums = witness_sums .* witness_signs;
end

rows = 2;
cols = 10;
raster = [];
one_line = [];


[sorted, idx] = sort(witness_sums, 'ascend');

for c = idx(1:cols)'
    display(witness_sums(c) / MMD);
    [~, idx_c] = sort(witnesses_Y.*(c_Y==c), 'ascend');
    %imagesc(reshape(Y(idx_c(1),:), 28, 28)');
    %drawnow;
    %pause;
    one_line = [one_line, reshape(Y(idx_c(1),:), 28, 28)'];
end

raster = [raster; one_line];
one_line = [];

[sorted, idx] = sort(witness_sums, 'descend');

for c = idx(1:cols)'
    display(witness_sums(c) / MMD);
    [~, idx_c] = sort(witnesses_X.*(c_X==c), 'descend');
%     imagesc(reshape(X(idx_c(1),:), 28, 28)');
%     drawnow;
%     pause;
    one_line = [one_line, reshape(X(idx_c(1),:), 28, 28)'];
end

raster = [raster; one_line];

h = figure('Position', [300, 300, 000+size(raster,2), 000+size(raster,1)]);
imagesc(-raster);
colormap(bone);
set(gca, 'YTick', []);
set(gca, 'XTick', []);
save2pdf( 'temp/witness_peaks.pdf', h, 600, true );

%% Plot some over represented digits - conditional dist

m = size(X_dr, 1);
n = size(Y_dr, 1);
t_X_dr = X_dr;
t_Y_dr = Y_dr;
K1 = rbf_dot(X_dr, t_X_dr, params.sig);
K2 = rbf_dot(Y_dr, t_X_dr, params.sig);
witness_X = sum(K1, 1)' / m - sum(K2, 1)' / n;
K1 = rbf_dot(X_dr, t_Y_dr, params.sig);
K2 = rbf_dot(Y_dr, t_Y_dr, params.sig);
witness_Y = sum(K1, 1)' / m - sum(K2, 1)' / n;

raster = [];
one_line = [];
for digit = 0:9;
    Y_i = Y(Y_labels==digit,:);
    witness_Y_i = witness_Y(Y_labels==digit,:);
    [~, i] = sort(witness_Y_i, 'ascend');
    one_line = [one_line, reshape(Y_i(i(1),:), 28, 28)'];
end
raster = [raster; one_line];
one_line = [];
for digit = 0:9;
    X_i = X(X_labels==digit,:);
    witness_X_i = witness_X(X_labels==digit,:);
    [~, i] = sort(witness_X_i, 'descend');
    one_line = [one_line, reshape(X_i(i(1),:), 28, 28)'];
end
raster = [raster; one_line];
h = figure('Position', [300, 300, 000+size(raster,2), 000+size(raster,1)]);
imagesc(-raster);
colormap(bone);
set(gca, 'YTick', []);
set(gca, 'XTick', []);
save2pdf('temp/cond.pdf', h, 600, true );