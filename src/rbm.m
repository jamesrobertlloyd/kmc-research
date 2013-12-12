%% Clear and load mmd tools

clear all;
addpath(genpath('./mmd'));

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

%% Load rbm fantasies

fantasies = csvread('../data/mnist/rbm-samples/images.csv');
labels = csvread('../data/mnist/rbm-samples/labels.csv');
num_images = 3000;

%% Load multi rbm fantasies

fantasies = csvread('../data/mnist/many-rbm-samples/images.csv');
labels = csvread('../data/mnist/many-rbm-samples/labels.csv');
num_images = 294;
           
%% Standardise digit data

test_digits = double(test_digits);
test_digits = test_digits / max(max(test_digits));
fantasies = fantasies / max(max(fantasies));

%% Randomize order of test digits

perm = randperm(size(test_digits, 1));
test_digits = test_digits(perm,:);

%% Display random fantasies

i = randi(num_images);
imagesc(reshape(fantasies(i,:), 28, 28)');
display(labels(i));

%% Extract digits

X = test_digits(1:num_images,:);
Y = fantasies(1:num_images,:);

%% Extract digits

X = test_digits(1:num_images,:);
Y = test_digits(num_images:(num_images+num_images-1),:);

%% Calculate some distances for reference

d1 = sqrt(sq_dist(X', X'));
d2 = sqrt(sq_dist(Y', Y'));
hist([d1(:);d2(:)]);

%% Perform MMD test

alpha = 0.05;
params.sig = -1;
params.shuff = 100;
[testStat,thresh,params] = mmdTestBoot_jl(X,Y,alpha,params);
testStat
thresh
params

testStat / thresh

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

% average_image = mean(X(i(1:200),:),1);
% imagesc(reshape(average_image, 28, 28)');
% pause;

%i = find(witness_X==max(witness_X));

%imagesc(reshape(X(i,:), 28, 28)');

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
i = find(witness_Y==min(witness_Y(standard_score_1(m+1:end) < -0.4)));
x = standard_score_1(i+m);
y = standard_witness(i+m);
imagesc([x-width,x+width],[y+width,y-width], reshape(-X(i,:), 28, 28)');
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
for dummy = 1:10;
    i = randi(m);
    x = standard_score_1(i);
    y = standard_witness(i);
    imagesc([x-width,x+width],[y+width,y-width], reshape(-X(i,:), 28, 28)');
end
for dummy = 1:10;
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