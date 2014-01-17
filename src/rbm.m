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

%% Perform PCA preprocessing

d = 2;
coeff = pca([X;Y]);
X = X * coeff(:,1:d);
Y = Y * coeff(:,1:d);
plot(X(:,1), X(:,2), 'go');
hold on;
plot(Y(:,1), Y(:,2), 'rx');
hold off;

%% Perform random projection preprocessing

d = 2;
coeff = randn(size(X,2));
X = X * coeff(:,1:d);
Y = Y * coeff(:,1:d);
plot(X(:,1), X(:,2), 'go');
hold on;
plot(Y(:,1), Y(:,2), 'rx');
hold off;

%% Calculate some distances for reference

d1 = sqrt(sq_dist(X', X'));
d2 = sqrt(sq_dist(Y', Y'));
hist([d1(:);d2(:)]);

%% Perform MMD test

alpha = 0.01;
%params.sig = -1;
params.sig = 2.5;
params.shuff = 100;
[testStat,thresh,params] = mmdTestBoot_jl(X,Y,alpha,params);
testStat
thresh
params

testStat / thresh

%% Find peaks of the witness function on fantasies

close all;

%ell = params.sig;
ell = 2;
x_opt_Y = zeros(size(Y));
witnesses_Y = zeros(size(Y, 1), 1);

for i = 1:size(Y,1)

    x = Y(i,:)';
    witness = rbf_witness(x, X, Y, ell);
    witnesses_Y(i) = witness;
    fprintf('\nwitness=%f\n', witness);

    % Checkgrad
    % options = optimoptions('fminunc','GradObj','on', ...
    %                        'DerivativeCheck', 'on', ...
    %                        'FinDiffType', 'central');
    % fminunc(@(x) rbf_witness(x, X, Y, ell), x, options);  

    if witness >=0 
        % Maximize
        x = minimize(x, @(x) neg_rbf_witness(x, X, Y, ell), -50);
    else
        x = minimize(x, @(x) rbf_witness(x, X, Y, ell), -50);
    end
    x_opt_Y(i,:) = x';
    %imagesc(reshape(x, 28, 28)');
    %drawnow;
    
    fprintf('\ni = %d\n', i);
    
end

%% Find peaks of the witness function on test

close all;

%ell = params.sig;
ell = 2;
x_opt_X = zeros(size(X));
witnesses_X = zeros(size(X, 1), 1);

for i = 1:size(X,1)

    x = X(i,:)';
    witness = rbf_witness(x, X, Y, ell);
    witnesses_X(i) = witness;
    fprintf('\nwitness=%f\n', witness);

%     % Checkgrad
%     options = optimoptions('fminunc','GradObj','on', ...
%                            'DerivativeCheck', 'on', ...
%                            'FinDiffType', 'central');
%     fminunc(@(x) neg_rbf_witness(x, X, Y, ell), x, options);  

    if witness >=0 
        % Maximize
        x = minimize(x, @(x) neg_rbf_witness(x, X, Y, ell), -50);
    else
        x = minimize(x, @(x) rbf_witness(x, X, Y, ell), -50);
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
c_XY = [c_X; c_Y];
witnesses_XY = [witnesses_X; witnesses_Y];
for c = 1:max_c
    % TODO - these need to be weighted if m <> n
    witness_sums(c) = sum(witnesses_X(c_X==c)) - sum(witnesses_Y(c_Y==c));
    if sum(witnesses_XY(c_XY==c)) < 0
        witness_sums(c) = -witness_sums(c);
    end
end

[sorted, idx] = sort(witness_sums, 'ascend');

for c = idx(1:10)'
    [~, idx_c] = sort(witnesses_Y.*(c_Y==c), 'ascend');
    imagesc(reshape(Y(idx_c(1),:), 28, 28)');
    drawnow;
    pause;
end

[sorted, idx] = sort(witness_sums, 'descend');

for c = idx(1:10)'
    [~, idx_c] = sort(witnesses_X.*(c_X==c), 'descend');
    imagesc(reshape(X(idx_c(1),:), 28, 28)');
    drawnow;
    pause;
end

%% Compute witness function and show (least) favourite images

%params.sig = 7;

m = size(X, 1);
n = size(Y, 1);
t_X = X;
t_Y = Y;
K1 = rbf_dot(X, t_X, params.sig);
K2 = rbf_dot(Y, t_X, params.sig);
witness_X = sum(K1, 1)' / m - sum(K2, 1)' / n;
K1 = rbf_dot(X, t_Y, params.sig);
K2 = rbf_dot(Y, t_Y, params.sig);
witness_Y = sum(K1, 1)' / m - sum(K2, 1)' / n;

%i = find(witness_Y<min(witness_Y)*0.8);

[~, i] = sort(witness_Y, 'ascend');

for j = 1:10
    imagesc(reshape(Y(i(j),:), 28, 28)');
    drawnow;
    pause;
end

[~, i] = sort(witness_X, 'ascend');

for j = 1:10
    imagesc(reshape(X(i(j),:), 28, 28)');
    drawnow;
    pause;
end

% Average image

% average_image = mean(Y(i(1:200),:),1);
% imagesc(reshape(average_image, 28, 28)');
% pause;

[~, i] = sort(witness_X, 'descend');

for j = 1:10
    imagesc(reshape(X(i(j),:), 28, 28)');
    drawnow;
    pause;
end

[~, i] = sort(witness_Y, 'descend');

for j = 1:10
    imagesc(reshape(Y(i(j),:), 28, 28)');
    drawnow;
    pause;
end

% average_image = mean(X(i(1:200),:),1);
% imagesc(reshape(average_image, 28, 28)');
% pause;

%i = find(witness_X==max(witness_X));

%imagesc(reshape(X(i,:), 28, 28)');

%% Plot some over represented images

%params.sig = 7;

m = size(X, 1);
n = size(Y, 1);
t_X = X;
t_Y = Y;
K1 = rbf_dot(X, t_X, params.sig);
K2 = rbf_dot(Y, t_X, params.sig);
witness_X = sum(K1, 1)' / m - sum(K2, 1)' / n;
K1 = rbf_dot(X, t_Y, params.sig);
K2 = rbf_dot(Y, t_Y, params.sig);
witness_Y = sum(K1, 1)' / m - sum(K2, 1)' / n;

raster = [];
one_line = [];
[~, i] = sort(witness_Y, 'ascend');
for j = 1:10
    one_line = [one_line, reshape(Y(i(j),:), 28, 28)'];
end
raster = [raster; one_line];
one_line = [];
[~, i] = sort(witness_X, 'descend');
for j = 1:10
    one_line = [one_line, reshape(X(i(j),:), 28, 28)'];
end
raster = [raster; one_line];
h = figure('Position', [300, 300, 000+size(raster,2), 000+size(raster,1)]);
imagesc(-raster);
colormap(bone);
set(gca, 'YTick', []);
set(gca, 'XTick', []);
save2pdf( 'samples.pdf', h, 600, true );

%% Plot some over represented digits - conditional dist

%params.sig = 7;

m = size(X, 1);
n = size(Y, 1);
t_X = X;
t_Y = Y;
K1 = rbf_dot(X, t_X, params.sig);
K2 = rbf_dot(Y, t_X, params.sig);
witness_X = sum(K1, 1)' / m - sum(K2, 1)' / n;
K1 = rbf_dot(X, t_Y, params.sig);
K2 = rbf_dot(Y, t_Y, params.sig);
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
save2pdf( 'samples.pdf', h, 600, true );

%% Plot some over represented digits - PCA

%params.sig = 7;

m = size(X, 1);
n = size(Y, 1);
t_X = X;
t_Y = Y;
K1 = rbf_dot(X, t_X, params.sig);
K2 = rbf_dot(Y, t_X, params.sig);
witness_X = sum(K1, 1)' / m - sum(K2, 1)' / n;
K1 = rbf_dot(X, t_Y, params.sig);
K2 = rbf_dot(Y, t_Y, params.sig);
witness_Y = sum(K1, 1)' / m - sum(K2, 1)' / n;

[coeff, score, latent] = pca([X;Y]);
standard_score_1 = score(:,1) / range(score(:,1));
standard_score_1 = standard_score_1(:,1) - min(standard_score_1) - 0.5;
standard_witness = [witness_X; witness_Y];
standard_images = [X; Y];

raster = [];
one_line = [];
for position = 1:10;
    upper = -0.5 + (position * 0.1);
    lower = upper - 0.1;
    idx = (standard_score_1 >= lower) & (standard_score_1 <= upper);
    image_i = standard_images(idx,:);
    witness_i = standard_witness(idx,:);
    [~, i] = sort(witness_i, 'ascend');
    one_line = [one_line, reshape(image_i(i(1),:), 28, 28)'];
end
raster = [raster; one_line];
one_line = [];
for position = 1:10;
    upper = -0.5 + (position * 0.1);
    lower = upper - 0.1;
    idx = (standard_score_1 >= lower) & (standard_score_1 <= upper);
    image_i = standard_images(idx,:);
    witness_i = standard_witness(idx,:);
    [~, i] = sort(witness_i, 'descend');
    one_line = [one_line, reshape(image_i(i(1),:), 28, 28)'];
end
raster = [raster; one_line];
h = figure('Position', [300, 300, 000+size(raster,2), 000+size(raster,1)]);
imagesc(-raster);
colormap(bone);
set(gca, 'YTick', []);
set(gca, 'XTick', []);
save2pdf( 'samples.pdf', h, 600, true );

%% Do PCA

m = size(X, 1);
[coeff, score, latent] = pca([X;Y]);
score = score - repmat(min(score), size(score, 1), 1);
score = score ./ repmat(range(score), size(score, 1), 1);
score = (score - 0.5) * 2;

%% Half finished plotting function

t_loc = (((fullfact([50,50])-0.5) / 25) - 1) * 1;
witness_loc = score(:,1:2);
witness = [witness_X; witness_Y];
K_Wt = rbf_dot(witness_loc, t_loc, 0.1);
t = sum(K_Wt .* repmat(witness, 1, size(t_loc, 1)), 1)';
t = (t - min(t)) / (max(t) - min(t));

plot(score(1:m,1), score(1:m,2), 'o');
hold on;
plot(score((m+1):end,1), score((m+1):end,2), 'ro');
imagesc([-1, 1], [-1, 1], reshape(t(end:-1:1), 50, 50));
colormap(bone);
for dummy = 1:40;
    i = randi(m);
    x = score(i,1);
    y = score(i,2);
    imagesc([x-width,x+width],[y+width,y-width], reshape(-2*X(i,:), 28, 28)');
end
for dummy = 1:40;
    i = randi(size(Y,1));
    x = score(i+m,1);
    y = score(i+m,2);
    imagesc([x-width,x+width],[y+width,y-width], reshape(-2*Y(i,:), 28, 28)');
end
plot(score(1:m,1), score(1:m,2), 'o');
plot(score((m+1):end,1), score((m+1):end,2), 'ro');
xlim([-1,1]);
ylim([-1,1]);
hold off;

%% Plot something

[coeff, score, latent] = pca([test_digits;Y]);
m_2 = size(test_digits, 1);
plot(score(1:m_2,1), score(1:m_2,2), 'o');
hold on;
plot(score((m_2+1):end,1), score((m_2+1):end,2), 'ro');
hold off;

%%

[coeff, score, latent] = pca([X;Y]);
plot3(score(1:m,1), score(1:m,2), witness_X, 'o');
hold on;
plot3(score((m+1):end,1), score((m+1):end,2), witness_Y, 'ro');
hold off;

%% 1d PCA - not useful mathematically - but pretty

[coeff, score, latent] = pca([X;Y]);
%plot(score(:,1), score(:,2), 'o');
%plot3(score(:,1), score(:,2), witness_X, 'o');
% plot3(score(1:m,1), score(1:m,2), witness_X, 'o');
% hold on;
% plot3(score((m+1):end,1), score((m+1):end,2), witness_Y, 'ro');
% hold off;
standard_score_1 = score(:,1) / range(score(:,1));
standard_score_1 = standard_score_1(:,1) - mean(standard_score_1);
standard_witness = [witness_X; witness_Y];
standard_witness = standard_witness(:,1) / range(standard_witness(:,1));
standard_witness = standard_witness(:,1) - mean(standard_witness);
plot(standard_score_1(1:m), standard_witness(1:m), 'o');
hold on;
plot(standard_score_1((m+1):end), standard_witness((m+1):end), 'ro');
i = find(witness_Y==min(witness_Y(standard_score_1(m+1:end) < 0.5)));
x = standard_score_1(i+m);
y = standard_witness(i+m);
width = 0.025;
imagesc([x-width,x+width],[y+width,y-width], reshape(-fantasies(i,:), 28, 28)');
colormap(bone);
i = find(witness_Y==min(witness_Y(standard_score_1(m+1:end) < 0.4)));
x = standard_score_1(i+m);
y = standard_witness(i+m);
width = 0.025;
imagesc([x-width,x+width],[y+width,y-width], reshape(-fantasies(i,:), 28, 28)');
i = find(witness_Y==min(witness_Y(standard_score_1(m+1:end) < 0.3)));
x = standard_score_1(i+m);
y = standard_witness(i+m);
width = 0.025;
imagesc([x-width,x+width],[y+width,y-width], reshape(-fantasies(i,:), 28, 28)');
i = find(witness_Y==min(witness_Y(standard_score_1(m+1:end) < 0.2)));
x = standard_score_1(i+m);
y = standard_witness(i+m);
imagesc([x-width,x+width],[y+width,y-width], reshape(-fantasies(i,:), 28, 28)');
i = find(witness_Y==min(witness_Y(standard_score_1(m+1:end) < 0.1)));
x = standard_score_1(i+m);
y = standard_witness(i+m);
imagesc([x-width,x+width],[y+width,y-width], reshape(-fantasies(i,:), 28, 28)');
i = find(witness_Y==min(witness_Y(standard_score_1(m+1:end) < 0.0)));
x = standard_score_1(i+m);
y = standard_witness(i+m);
imagesc([x-width,x+width],[y+width,y-width], reshape(-fantasies(i,:), 28, 28)');
i = find(witness_Y==min(witness_Y(standard_score_1(m+1:end) < -0.1)));
x = standard_score_1(i+m);
y = standard_witness(i+m);
imagesc([x-width,x+width],[y+width,y-width], reshape(-fantasies(i,:), 28, 28)');
i = find(witness_Y==min(witness_Y(standard_score_1(m+1:end) < -0.2)));
x = standard_score_1(i+m);
y = standard_witness(i+m);
imagesc([x-width,x+width],[y+width,y-width], reshape(-fantasies(i,:), 28, 28)');
i = find(witness_Y==min(witness_Y(standard_score_1(m+1:end) < -0.3)));
x = standard_score_1(i+m);
y = standard_witness(i+m);
imagesc([x-width,x+width],[y+width,y-width], reshape(-fantasies(i,:), 28, 28)');
% i = find(witness_Y==min(witness_Y(standard_score_1(m+1:end) < -0.4)));
% x = standard_score_1(i+m);
% y = standard_witness(i+m);
% imagesc([x-width,x+width],[y+width,y-width], reshape(-fantasies(i,:), 28, 28)');
i = find(witness_X==max(witness_X(standard_score_1(1:m) > -0.5)));
x = standard_score_1(i);
y = standard_witness(i);
imagesc([x-width,x+width],[y+width,y-width], reshape(-X(i,:), 28, 28)');
i = find(witness_X==max(witness_X(standard_score_1(1:m) > -0.4)));
x = standard_score_1(i);
y = standard_witness(i);
imagesc([x-width,x+width],[y+width,y-width], reshape(-X(i,:), 28, 28)');
i = find(witness_X==max(witness_X(standard_score_1(1:m) > -0.3)));
x = standard_score_1(i);
y = standard_witness(i);
imagesc([x-width,x+width],[y+width,y-width], reshape(-X(i,:), 28, 28)');
i = find(witness_X==max(witness_X(standard_score_1(1:m) > -0.2)));
x = standard_score_1(i);
y = standard_witness(i);
imagesc([x-width,x+width],[y+width,y-width], reshape(-X(i,:), 28, 28)');
i = find(witness_X==max(witness_X(standard_score_1(1:m) > -0.1)));
x = standard_score_1(i);
y = standard_witness(i);
imagesc([x-width,x+width],[y+width,y-width], reshape(-X(i,:), 28, 28)');
i = find(witness_X==max(witness_X(standard_score_1(1:m) > -0.0)));
x = standard_score_1(i);
y = standard_witness(i);
imagesc([x-width,x+width],[y+width,y-width], reshape(-X(i,:), 28, 28)');
i = find(witness_X==max(witness_X(standard_score_1(1:m) > 0.1)));
x = standard_score_1(i);
y = standard_witness(i);
imagesc([x-width,x+width],[y+width,y-width], reshape(-X(i,:), 28, 28)');
i = find(witness_X==max(witness_X(standard_score_1(1:m) > 0.2)));
x = standard_score_1(i);
y = standard_witness(i);
imagesc([x-width,x+width],[y+width,y-width], reshape(-X(i,:), 28, 28)');
i = find(witness_X==max(witness_X(standard_score_1(1:m) > 0.3)));
x = standard_score_1(i);
y = standard_witness(i);
imagesc([x-width,x+width],[y+width,y-width], reshape(-X(i,:), 28, 28)');
i = find(witness_X==max(witness_X(standard_score_1(1:m) > 0.4)));
x = standard_score_1(i);
y = standard_witness(i);
imagesc([x-width,x+width],[y+width,y-width], reshape(-X(i,:), 28, 28)');
for dummy = 1:20;
    i = randi(m);
    x = standard_score_1(i);
    y = standard_witness(i);
    imagesc([x-width,x+width],[y+width,y-width], reshape(-X(i,:), 28, 28)');
end
for dummy = 1:20;
    i = randi(size(Y,1));
    x = standard_score_1(i+m);
    y = standard_witness(i+m);
    imagesc([x-width,x+width],[y+width,y-width], reshape(-Y(i,:), 28, 28)');
end
plot(standard_score_1(1:m), standard_witness(1:m), 'o');
plot(standard_score_1((m+1):end), standard_witness((m+1):end), 'ro');
hold off;
% plot3(score(1:m,1), score(1:m,2), score(1:m,3), 'o');
% hold on;
% plot3(score((m+1):end,1), score((m+1):end,2), score((m+1):end,3), 'ro');
% hold off;
% figure();
% hold on;
% indices = 
% plot(score(:,1), score(:,2), 'o');
% hold off;