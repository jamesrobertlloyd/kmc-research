%% Clear and load mmd tools and gpml

% clear all;
addpath(genpath('../mmd'));
addpath(genpath('../util'));
addpath(genpath('../gpml'));

%% Setup

folders = {'SE', 'TCI', 'SP', 'ABCD'};
folder = folders{1};

files = dir(strcat(folder, '/*.mat'));
file = files(9);

fig_title = 'Gas production';

init_rand(1);

%% Load data

load(strcat(folder, '/', file.name));

X_data = double(Xtest);
y_data = double(ytest);

X_post = double(Xtest);
y_post = ymu + sqrt(ys2) .* randn(size(ys2));

%% Plot

h = figure;

% plot(X_data, y_data, 'bo');
% hold on;
% % plot(X_data, y_post, 'ro');
% hold off;
[dummy, idx] = sort(X_data, 'ascend');
mean_var_plot(X_data, y_data, X_data(idx), ymu(idx), 2*sqrt(ys2(idx)));
xlabel('x');
ylabel('y');
xlim([min(X_data)-0, max(X_data)+0]);
ylim([min(y_data)-0, max(y_data)+0]);
title(fig_title);

save2pdf(['temp/' file.name '-data.pdf'], h, 900, true);

%% Get ready for two sample test

A = [X_data, y_data];
B = [X_post, y_post];

%% Standardise data

std_A = std(A);

B = B ./ repmat(std_A, size(B, 1), 1);
A = A ./ repmat(std_A, size(A, 1), 1);

%% Calculate some distances for reference

d1 = sqrt(sq_dist(A', A'));
d2 = sqrt(sq_dist(B', B'));

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
%             display(ell);
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
params.sig = 1 * best_ell;
%         display(params.sig);

%% Compute witness function

increase = 0.0;

if size(A,2) == 2
    m = size(A, 1);
    n = size(B, 1);
    t = (((fullfact([200,200])-0.5) / 200) - 0) * 1;
    t = t .* ((1+increase) * repmat(range([A]), size(t,1), 1));
    t = t + repmat(min([A]) - (increase/2)*range([A]), size(t,1), 1);
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
    save2pdf(['temp/' file.name '-witness.pdf'], h, 900, true);
    hold off;
end