%% Clear and load mmd tools

clear all;
addpath(genpath('./mmd'));
addpath(genpath('./util'));

%% Load newcomb data

Y = csvread('../data/newcomb/newcomb.csv', 1, 1);

%% Fit gaussian to it

mu = mean(Y);
sd = std(Y);

%% Fit gaussian ignoring outliers

mu = mean(Y(Y > 24.81));
sd = std(Y(Y > 24.81));

%% Sample replicate data

X = mu + sd * randn(size(Y));
%X = mu + sd * randn(1000,1);

%% Calculate some distances for reference

d1 = sqrt(sq_dist(X', X'));
d2 = sqrt(sq_dist(Y', Y'));
hist([d1(:);d2(:)], 50);

%% Perform MMD test

alpha = 0.05;
params.sig = -1;
%params.sig = 0.0034;
%params.sig = 0.001;
params.shuff = 1000;
[testStat,thresh,params] = mmdTestBoot_jl(X,Y,alpha,params);
testStat
thresh
params

testStat / thresh

%% Compute witness function

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
t = linspace(min(Y)-0.05, max(Y)+0.05, 1000)';
K1 = rbf_dot(X, t, params.sig);
K2 = rbf_dot(Y, t, params.sig);
witness = sum(K1, 1)' / m - sum(K2, 1)' / n;

hist(Y, 500);
hold on;
plot(t, witness * 10, 'b', 'linewidth', 2);
plot(t, 0.1*exp(-(t-mu).^2./(2*sd^2)) / sqrt(2*pi*sd^2), 'r');
hold off;