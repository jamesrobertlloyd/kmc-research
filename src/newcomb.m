%% Clear and load mmd tools

clear all;
addpath(genpath('./mmd'));
addpath(genpath('./util'));

%% Load newcomb data

Y = csvread('../data/newcomb/newcomb.csv', 1, 1);
% This is the scaling that appears in Bayesian data analysis
Y = (Y - 24.8)*1000;

%% Plot data

h = figure();
hold on;
hist(Y, 35);
xlabel('Deviations from 24,800 nanoseconds', 'FontSize', 15)
ylabel('Count', 'FontSize', 15);
hold off;
save2pdf('newcomb_hist.pdf', h, 600, true);

%% Fit gaussian to it

mu = mean(Y);
sd = std(Y);

%% Fit gaussian ignoring outliers

mu = mean(Y(Y > 0));
sd = std(Y(Y > 0));

%% Sample replicate data - Gaussian

%X = mu + sd * randn(size(Y));
X = mu + sd * randn(1000,1);

%% Sample replicate data - t

v = 2;
X = mu + sd * trnd(v,1000,1) / std(trnd(v,10000,1));

%% Calculate some distances for reference

d1 = sqrt(sq_dist(X', X'));
d2 = sqrt(sq_dist(Y', Y'));
hist([d1(:);d2(:)], 50);

%% Perform MMD test

alpha = 0.05;
params.sig = -1;
params.shuff = 1000;
[testStat,thresh,params,p] = mmdTestBoot_jl(X,Y,alpha,params);
testStat
thresh
params

testStat / thresh
p

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
t = linspace(-50, 60, 1000)';
K1 = rbf_dot(X, t, params.sig);
K2 = rbf_dot(Y, t, params.sig);
witness = sum(K1, 1)' / m - sum(K2, 1)' / n;

[hist_N,hist_X] = hist(Y, 35);
h = figure();
hold on;
bar(hist_X,hist_N/(66*(hist_X(2)-hist_X(1))));
xlim([-50,60]);
[AX,H1,H2] = plotyy(t,exp(-(t-mu).^2./(2*sd^2)) / sqrt(2*pi*sd^2),t,witness,'plot');
set(get(AX(1),'Ylabel'),'String','Density estimate','FontSize',15) 
set(get(AX(2),'Ylabel'),'String','Witness function','FontSize',15)
set(AX(1), 'ylim', [-0.15,0.15]) 
set(AX(2), 'ylim', [-0.25,0.25]) 
set(H1,'LineStyle','-')
set(H1,'Color','r')
set(H2,'LineStyle','--')
set(H1,'LineWidth',4)
set(H2,'LineWidth',4)
xlabel('Deviations from 24,800 nanseconds','FontSize',15)
hold off
save2pdf('newcomb_witness_2.pdf', h, 600, true);