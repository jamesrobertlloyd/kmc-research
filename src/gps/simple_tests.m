%% Clear and load mmd tools and gpml

clear all;
addpath(genpath('../mmd'));
addpath(genpath('../util'));
addpath(genpath('../gpml'));

init_rand(1);

%% Generate some data with a discontinuity

% init_rand(1);
% 
% N = 100;
% 
% X = linspace(0, 1, N)';
% K = covSEiso([-2,0], X);
% K = K + 0.1 * eye(size(K));
% 
% y = chol(K)' * randn(N, 1);
% y(X > 0.5) = 20 + y(X > 0.5);
% 
% plot(X, y, 'o');

%% Generate some data with heteroscedasticity

% init_rand(1);
% 
% N = 500;
% 
% X = linspace(0, 1, N)';
% K = covSEiso([-2,0], X);
% K = K + 1 * diag((X-0.5).^2);
% 
% y = chol(K)' * randn(N, 1);
% 
% plot(X, y, 'o');

%% Load HadCRUT

% load 'HadCRUT-4-2-0-0-monthly-ns-avg-median-only'
% 
% plot(X, y, 'o');
% 
% % Subsample
% 
% % sub_sample = randsample(1:numel(y), 500);
% % X = X(sub_sample,:);
% % y = y(sub_sample);

%% Load concrete

% load 'concrete'
% 
% X = X ./ repmat(std(X), size(X, 1), 1);
% y = y ./ repmat(std(y), size(y, 1), 1);
% 
% % Subsample
% % sub_sample = randsample(1:numel(y), 500);
% % X = X(sub_sample,:);
% % y = y(sub_sample);

%% Load housing

% load 'housing'
% 
% X = X ./ repmat(std(X), size(X, 1), 1);
% y = y ./ repmat(std(y), size(y, 1), 1);
% 
% % Subsample
% % sub_sample = randsample(1:numel(y), 500);
% % X = X(sub_sample,:);
% % y = y(sub_sample);

%% Load solar

load '02-solar'

X = X - repmat(mean(X), size(X, 1), 1);
y = y - repmat(mean(y), size(y, 1), 1);

X = X ./ repmat(std(X), size(X, 1), 1);
y = y ./ repmat(std(y), size(y, 1), 1);

% Subsample
% sub_sample = randsample(1:numel(y), 500);
% X = X(sub_sample,:);
% y = y(sub_sample);

%% Load mauna

% load '03-mauna'
% 
% X = X - repmat(mean(X), size(X, 1), 1);
% y = y - repmat(mean(y), size(y, 1), 1);
% 
% X = X ./ repmat(std(X), size(X, 1), 1);
% y = y ./ repmat(std(y), size(y, 1), 1);

%% Load internet

% load '06-internet'
% 
% X = X - repmat(mean(X), size(X, 1), 1);
% y = y - repmat(mean(y), size(y, 1), 1);
% 
% X = X ./ repmat(std(X), size(X, 1), 1);
% y = y ./ repmat(std(y), size(y, 1), 1);

%% Load call centre

% load '07-call-centre'
% 
% X = X - repmat(mean(X), size(X, 1), 1);
% y = y - repmat(mean(y), size(y, 1), 1);
% 
% X = X ./ repmat(std(X), size(X, 1), 1);
% y = y ./ repmat(std(y), size(y, 1), 1);

%% Load gas

% load '09-gas-production'
% 
% X = X - repmat(mean(X), size(X, 1), 1);
% y = y - repmat(mean(y), size(y, 1), 1);
% 
% X = X ./ repmat(std(X), size(X, 1), 1);
% y = y ./ repmat(std(y), size(y, 1), 1);

%% Load unemployment

% load '11-unemployment'
% 
% X = X - repmat(mean(X), size(X, 1), 1);
% y = y - repmat(mean(y), size(y, 1), 1);
% 
% X = X ./ repmat(std(X), size(X, 1), 1);
% y = y ./ repmat(std(y), size(y, 1), 1);

%% Load wages

% load '13-wages'
% 
% X = X - repmat(mean(X), size(X, 1), 1);
% y = y - repmat(mean(y), size(y, 1), 1);
% 
% X = X ./ repmat(std(X), size(X, 1), 1);
% y = y ./ repmat(std(y), size(y, 1), 1);

%% Fit a GP to this

% if numel(y) > 500
%     % Subset of data approx
%     sub_sample = randsample(1:numel(y), 500);
%     X_train = X(sub_sample,:);
%     y_train = y(sub_sample);
% else
%     X_train = X;
%     y_train = y;
% end
% 
% hyp.cov = [0,0];
% hyp.mean = [];
% hyp.lik = 0;
% 
% cov_fn = @covSEiso;
% mean_fn = @meanZero;
% lik_fn = @likGauss;
% 
% inf = @infExact;
% 
% hyp_opt = minimize(hyp, @gp, -500, inf, mean_fn, cov_fn, lik_fn, X_train, y_train);

%% Fit a spectral mixture

if numel(y) > 500
    % Subset of data approx
    sub_sample = randsample(1:numel(y), 500);
    X_train = X(sub_sample,:);
    y_train = y(sub_sample);
else
    X_train = X;
    y_train = y;
end

hyp.cov = randn(1,5*4);
hyp.mean = [];
hyp.lik = 0;

cov_fn = {@covSum, {{@covProd, {@covCos, @covSEiso}}, ...
                    {@covProd, {@covCos, @covSEiso}}, ...
                    {@covProd, {@covCos, @covSEiso}}, ...
                    {@covProd, {@covCos, @covSEiso}}, ...
                    {@covProd, {@covCos, @covSEiso}}}};
mean_fn = @meanZero;
lik_fn = @likGauss;

inf = @infExact;

hyp_opt = minimize(hyp, @gp, -200, inf, mean_fn, cov_fn, lik_fn, X_train, y_train);

%% Fit SEard to this

% if numel(y) > 500
%     % Subset of data approx
%     sub_sample = randsample(1:numel(y), 500);
%     X_train = X(sub_sample,:);
%     y_train = y(sub_sample);
% else
%     X_train = X;
%     y_train = y;
% end
% 
% hyp.cov = zeros(1,size(X_train,2)+1);
% hyp.mean = [];
% hyp.lik = 0;
% 
% cov_fn = @covSEard;
% mean_fn = @meanZero;
% lik_fn = @likGauss;
% 
% inf = @infExact;
% 
% hyp_opt = minimize(hyp, @gp, -500, inf, mean_fn, cov_fn, lik_fn, X_train, y_train);

%% Fit a GP to this using median lengthscale heuristic

% if numel(y) > 2500
%     % Subset of data approx
%     sub_sample = randsample(1:numel(y), 500);
%     X_train = X(sub_sample,:);
%     y_train = y(sub_sample);
% else
%     X_train = X;
%     y_train = y;
% end
% 
% distances = sqrt(sq_dist(X_train', X_train'));
% log_ell = log(median(distances(distances>0)));
% 
% hyp.cov = [0];
% hyp.mean = [];
% hyp.lik = 0;
% 
% cov_fn = {@covSEisofixed, log_ell}; % Create a covScale which takes K as input
% mean_fn = @meanZero;
% lik_fn = @likGauss;
% 
% inf = @infExact;
% 
% hyp_opt = minimize(hyp, @gp, -500, inf, mean_fn, cov_fn, lik_fn, X_train, y_train);

%% Sample from GP - compare to data graphically

[ymu, ys2, ~, ~] = gp(hyp_opt, inf, mean_fn, cov_fn, lik_fn, X, y, X);

X_post = [];
y_post = [];

for i = 1:1
  X_post = [X_post; X]; %#ok<AGROW>
  y_post = [y_post; ymu + sqrt(ys2) .* randn(size(ys2))]; %#ok<AGROW>
end


plot(X_post, y_post, 'ro'); hold on;
plot(X, y, 'go'); hold off;

%% Get ready for two sample test

A = [X, y];
B = [X_post, y_post];

%% Standardise data

A = A ./ repmat(std(A), size(A, 1), 1);
B = B ./ repmat(std(B), size(B, 1), 1);

%% Calculate some distances for reference

d1 = sqrt(sq_dist(A', A'));
d2 = sqrt(sq_dist(B', B'));
C = [A;B];  %aggregate the sample
d3 = sq_dist(C', C');
% figure;
% hist([d1(:);d2(:)]);

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
m = size(A, 1);
n = size(B, 1);
d = size(A, 2);
A_perm = A(randperm(m),:);
B_perm = B(randperm(n),:);
X_f_train = cell(folds,1);
X_f_test = cell(folds,1);
Y_f_train = cell(folds,1);
Y_f_test = cell(folds,1);
for fold = 1:folds
    if fold == 1
        X_f_train{fold} = A_perm(floor(fold*m/folds):end,:);
        X_f_test{fold} = A_perm(1:(floor(fold*m/folds)-1),:);
        Y_f_train{fold} = B_perm(floor(fold*n/folds):end,:);
        Y_f_test{fold} = B_perm(1:(floor(fold*n/folds)-1),:);
    elseif fold == folds
        X_f_train{fold} = A_perm(1:floor((fold-1)*m/folds),:);
        X_f_test{fold} = A_perm(floor((fold-1)*m/folds + 1):end,:);
        Y_f_train{fold} = B_perm(1:floor((fold-1)*n/folds),:);
        Y_f_test{fold} = B_perm(floor((fold-1)*m/folds + 1):end,:);
    else
        X_f_train{fold} = [A_perm(1:floor((fold-1)*m/folds),:);
                           A_perm(floor((fold)*m/folds+1):end,:)];
        X_f_test{fold} = A_perm(floor((fold-1)*m/folds + 1):floor((fold)*m/folds),:);
        Y_f_train{fold} = [B_perm(1:floor((fold-1)*n/folds),:);
                           B_perm(floor((fold)*n/folds+1):end,:)];
        Y_f_test{fold} = B_perm(floor((fold-1)*n/folds + 1):floor((fold)*n/folds),:);
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
[testStat,thresh,params,p] = mmdTestBoot_jl(A,B,alpha,params);
display(p);
%pause;

%% Expand lengthscale

while p < 0.05
    params.sig = params.sig * 1.1;
    [testStat,thresh,params,p] = mmdTestBoot_jl(A,B,alpha,params);
    display(p);
end
params.sig = params.sig / 1.1;

%% Compute witness function in 2d

if size(A,2) == 2
    m = size(A, 1);
    n = size(B, 1);
    t = (((fullfact([200,200])-0.5) / 200) - 0) * 1;
    t = t .* (1.4 * repmat(range([A; B]), size(t,1), 1));
    t = t + repmat(min([A; B]) - 0.2*range([A; B]), size(t,1), 1);
    K1 = rbf_dot(A, t, params.sig);
    K2 = rbf_dot(B, t, params.sig);
    witness = sum(K1, 1)' / m - sum(K2, 1)' / n;
    %plot3(t(:,1), t(:,2), witness, 'bo');
    %hold on;
    %plot3(B(:,1), B(:,2), repmat(max(max(witness)), size(B)), 'ro');
    reshaped = reshape(witness, 200, 200)';

    h = figure;
    imagesc(reshaped(end:-1:1,:));
    colorbar;
    %save2pdf('temp/witness.pdf', h, 600, true );
    %hold off;
end