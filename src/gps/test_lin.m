%% Clear and load mmd tools and gpml

% clear all;
addpath(genpath('../mmd'));
addpath(genpath('../util'));
addpath(genpath('../gpml'));

%% Setup

p_values = nan(13, 1);

residuals = false;

data_i = 0;

files = dir(strcat('SE', '/*.mat'));
    
%% Loop over files

for file = files'
    
    %% Init

    init_rand(1);

    data_i = data_i + 1;

    %% Load data
    load(strcat('SE', '/', file.name));

    X_data = double(Xtest);
    y_data = double(ytest);
    
    %% Fit linear regression
    
    coef = polyfit(double(X),double(y),1);
    ymu = double(Xtest)*coef(1) + coef(2);
    ys2 = var(double(y) - double(X)*coef(1) - coef(2));
    
    %% Sample from posterior

    X_post = double(Xtest);
    y_post = ymu + sqrt(ys2) .* randn(size(ymu));

    if residuals
        y_data = y_data - ymu;
        y_post = y_post - ymu;
    end

    %% Plot

    plot(X_post, y_post, 'ro'); hold on;
    plot(X_data, y_data, 'go'); hold off;

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
    params.sig = best_ell;
%         display(params.sig);

    %% Calculate MMD statistic null distribution

    params.shuff = 1000;
    MMDs = zeros(params.shuff,1);
    for b = 1:params.shuff
        % Draw two samples of data
        y_post_1 = ymu + sqrt(ys2) .* randn(size(ymu));
        y_post_2 = ymu + sqrt(ys2) .* randn(size(ymu));
        if residuals
            y_post_1 = y_post_1 - ymu;
            y_post_2 = y_post_2 - ymu;
        end
        C = [X_post, y_post_1];
        D = [X_post, y_post_2];
        C = C ./ repmat(std_A, size(C, 1), 1);
        D = D ./ repmat(std_A, size(D, 1), 1);
        % Compute MMD
        m=size(C,1);
        n=size(D,1);
        K = rbf_dot(C,C,params.sig);
        L = rbf_dot(D,D,params.sig);
        KL = rbf_dot(C,D,params.sig);
        testStat = (1/m^2) * sum(sum(K)) - (2 / (m * n)) * sum(sum(KL)) + ...
                   (1/n^2) * sum(sum(L));
        % Save
        MMDs(b) = testStat;
    end

    %% Calculate MMD stat and p-value

    m=size(A,1);
    n=size(B,1);
    K = rbf_dot(A,A,params.sig);
    L = rbf_dot(B,B,params.sig);
    KL = rbf_dot(A,B,params.sig);
    testStat = (1/m^2) * sum(sum(K)) - (2 / (m * n)) * sum(sum(KL)) + ...
               (1/n^2) * sum(sum(L));

    p = sum(testStat < MMDs) / length(MMDs);

    %% Save and display

    p_values(data_i) = p;

    display(p_values);

end