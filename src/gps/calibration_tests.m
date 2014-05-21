%% Clear and load mmd tools and gpml

% clear all;
addpath(genpath('../mmd'));
addpath(genpath('../util'));
addpath(genpath('../gpml'));

init_rand(2);

repeats = 100;
p_values = zeros(repeats, 1);

%% Main loop

for main_iter = 1:repeats

    %% Generate some data

    N = 50;

    X = linspace(0, 1, N)';
    K = covSEiso([0,0], X);
    K = K + 0.1 * eye(size(K));

    y = chol(K)' * randn(N, 1);

    plot(X, y, 'o');

    %% Fit a GP to this

    if numel(y) > 500
        % Subset of data approx
        sub_sample = randsample(1:numel(y), 500);
        X_train = X(sub_sample,:);
        y_train = y(sub_sample);
    else
        X_train = X;
        y_train = y;
    end

    hyp.cov = [-2,0];
    hyp.mean = [];
    hyp.lik = 0;

    cov_fn = @covSEiso;
    mean_fn = @meanZero;
    lik_fn = @likGauss;

    inf = @infExact;

    hyp_opt = minimize(hyp, @gp, -100, inf, mean_fn, cov_fn, lik_fn, X_train, y_train);

    %% Sample from GP - compare to data graphically

    % This version uses X as given

%     [ymu, ys2, ~, ~] = gp(hyp_opt, inf, mean_fn, cov_fn, lik_fn, X, y, X);
% 
%     X_post = [];
%     y_post = [];
% 
%     for i = 1:1
%       X_post = [X_post; X]; %#ok<AGROW>
%       y_post = [y_post; ymu + sqrt(ys2) .* randn(size(ys2))]; %#ok<AGROW>
%     end
% 
%     X_data = X;
%     y_data = y;
% 
%     plot(X_post, y_post, 'ro'); hold on;
%     plot(X, y, 'go'); hold off;

    %% Sample from GP - compare to data graphically

    % This version partitions into training and test
    
    rand_indices = randsample(length(X), length(X), false);
    X_train = X(rand_indices(1:(N/2)));
    y_train = y(rand_indices(1:(N/2)));
    X_test = X(rand_indices((N/2 + 1):end));
    y_test = y(rand_indices((N/2 + 1):end));

    [ymu, ys2, ~, ~] = gp(hyp_opt, inf, mean_fn, cov_fn, lik_fn, X_train, y_train, X_test);

    X_post = [];
    y_post = [];

    for i = 1:1
      X_post = [X_post; X_test]; %#ok<AGROW>
      y_post = [y_post; ymu + sqrt(ys2) .* randn(size(ys2))]; %#ok<AGROW>
    end

    X_data = X_test;
    y_data = y_test;

    plot(X_post, y_post, 'ro'); hold on;
    plot(X_data, y_data, 'go'); hold off;

    %% Sample from GP - compare to data graphically

    % This version uses with replacement bootstrap X

%     X_post = randsample(X,length(X), true);
% 
%     [ymu, ys2, ~, ~] = gp(hyp_opt, inf, mean_fn, cov_fn, lik_fn, X, y, X_post);
% 
%     y_post = ymu + sqrt(ys2) .* randn(size(ys2));
% 
%     rand_indices = randsample(length(X), length(X), true);
%     X_data = X(rand_indices);
%     y_data = y(rand_indices);
% 
%     plot(X_post, y_post, 'ro'); hold on;
%     plot(X_data, y_data, 'go'); hold off;

    %% Sample from GP - compare to data graphically

    % This version uses subsampling bootstrap

%     X_post = randsample(X,length(X)/5, true);
% 
%     [ymu, ys2, ~, ~] = gp(hyp_opt, inf, mean_fn, cov_fn, lik_fn, X, y, X_post);
% 
%     y_post = ymu + sqrt(ys2) .* randn(size(ys2));
% 
%     rand_indices = randsample(length(X), 5*length(X), true);
%     X_data = X(rand_indices);
%     y_data = y(rand_indices);
% 
%     plot(X_post, y_post, 'ro'); hold on;
%     plot(X_data, y_data, 'go'); hold off;

    %% Sample from GP - compare to data graphically

    % This version uses subsampling bootstrap - subsampling actual data

%     rand_indices = randsample(length(X), length(X)/2, true);
%     X_post = X(rand_indices);
%     y_post = y(rand_indices);
% 
%     rand_indices = randsample(length(X), length(X)/2, true);
%     X_data = X(rand_indices);
%     y_data = y(rand_indices);
% 
%     plot(X_post, y_post, 'ro'); hold on;
%     plot(X_data, y_data, 'go'); hold off;

    %% Sample from GP - compare to data graphically

    % This version uses KDE bootstrap

%     bandwidth = 0.1;
%     
%     rand_indices = randsample(length(X), length(X), true);
%     X_post = X(rand_indices) + bandwidth * randn(size(X));
% 
%     [ymu, ys2, ~, ~] = gp(hyp_opt, inf, mean_fn, cov_fn, lik_fn, X, y, X_post);
% 
%     y_post = ymu + sqrt(ys2) .* randn(size(ys2));
% 
%     rand_indices = randsample(length(X), length(X), true);
%     
%     X_data = X(rand_indices) + bandwidth * randn(size(X));
%     y_data = y(rand_indices) + bandwidth * randn(size(y));
% 
%     plot(X_post, y_post, 'ro'); hold on;
%     plot(X_data, y_data, 'go'); hold off;

    %% Sample from GP - compare to data graphically

    % This version uses random 50 / 50 partition
    
%     rand_indices = randsample(length(X), length(X), false);
%     X_post = X(rand_indices(1:(N/2)));
% 
%     [ymu, ys2, ~, ~] = gp(hyp_opt, inf, mean_fn, cov_fn, lik_fn, X, y, X_post);
% 
%     y_post = ymu + sqrt(ys2) .* randn(size(ys2));
%     
%     X_data = X(rand_indices((N/2 + 1):end));
%     y_data = y(rand_indices((N/2 + 1):end));
% 
%     plot(X_post, y_post, 'ro'); hold on;
%     plot(X_data, y_data, 'go'); hold off;

    %% Check
    
%     X_data = randn(size(X));
%     y_data = randn(size(y));
%     X_post = randn(size(X));
%     y_post = randn(size(y));

    %% Get ready for two sample test

    A = [X_data, y_data];
    B = [X_post, y_post];

    %% Standardise data

    B = B ./ repmat(std(A), size(B, 1), 1);
    A = A ./ repmat(std(A), size(A, 1), 1);

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
    display(params.sig);

    %% Perform MMD test

    alpha = 0.05;
    params.shuff = 100;
%     [testStat,thresh,params,p] = mmdTestBoot_jl(A,B,alpha,params);
    [testStat,thresh,params,p] = mmdTestBoot_strat_jl(A,B,alpha,params);
    display(p);
    %pause;
    
    %% Record p

    p_values(main_iter) = p;

end

hist(p_values);
