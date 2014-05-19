%% Clear and load mmd tools and gpml

clear all;
addpath(genpath('../mmd'));
addpath(genpath('../util'));
addpath(genpath('../gpml'));

init_rand(1);

%% Setup

filenames = {'01-airline', ...
             '02-solar', ...
             '03-mauna', ...
             '04-wheat', ...
             '05-temperature', ...
             '06-internet', ...
             '07-call-centre', ...
             '08-radio', ...
             '09-gas-production', ...
             '10-sulphuric', ...
             '11-unemployment', ...
             '12-births', ...
             '13-wages'};
         
p_values = zeros(numel(filenames), 1);

%% Loop

for main_loop_i = 1:numel(filenames)
    
    filename = filenames{main_loop_i};
    load(filename);

    %% Standardise data

    X = X - repmat(mean(X), size(X, 1), 1);
    y = y - repmat(mean(y), size(y, 1), 1);

    X = X ./ repmat(std(X), size(X, 1), 1);
    y = y ./ repmat(std(y), size(y, 1), 1);

    %% Fit a GP to this

    if numel(y) > 1000
        % Subset of data approx
        sub_sample = randsample(1:numel(y), 1000);
        X_train = X(sub_sample,:);
        y_train = y(sub_sample);
    else
        X_train = X;
        y_train = y;
    end

    hyp.cov = [0,0];
    hyp.mean = [];
    hyp.lik = 0;

    cov_fn = @covSEiso;
    mean_fn = @meanZero;
    lik_fn = @likGauss;

    inf = @infExact;

    hyp_opt = minimize(hyp, @gp, -500, inf, mean_fn, cov_fn, lik_fn, X_train, y_train);

    %% Sample from GP - compare to data graphically

    X_post = randsample(X,length(X), true);

    [ymu, ys2, ~, ~] = gp(hyp_opt, inf, mean_fn, cov_fn, lik_fn, X, y, X_post);

    y_post = ymu + sqrt(ys2) .* randn(size(ys2));

    rand_indices = randsample(length(X), length(X), true);
    X_data = X(rand_indices);
    y_data = y(rand_indices);

    plot(X_post, y_post, 'ro'); hold on;
    plot(X_data, y_data, 'go'); hold off;

    %% Get ready for two sample test

    A = [X_data, y_data];
    B = [X_post, y_post];

    %% Standardise data

    B = B ./ repmat(std(A), size(A, 1), 1);
    A = A ./ repmat(std(A), size(A, 1), 1);

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
%         display(ell);
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

%     display(params.sig);

    %% Perform MMD test

    alpha = 0.05;
    params.shuff = 1000;
    [testStat,thresh,params,p] = mmdTestBoot_jl(A,B,alpha,params);
    display(p);
    %pause;
    
    p_values(main_loop_i) = p;

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
        save2pdf(['temp/' filename '-witness.pdf'], h, 900, true);
        hold off;
    end
    
end

display(p_values);