%% Clear and load mmd tools

clear all;
close all;
addpath(genpath('./mmd'));
addpath(genpath('./util'));

%% Init PRNGs

seed = 1;
init_rand(seed);

%% Create some high dimensional mixture of Gaussians

n_per_cluster = 250;
d = 10;
effective_d = 4;
n_clusters = 5;
cluster_sd = 0.2;

basis = randn(d, effective_d);

cluster_centres = (basis * randn(effective_d, n_clusters))';

Y = [];
for i = 1:n_clusters
    for j = 1:n_per_cluster
        if i < n_clusters
            % Add Gaussian 
            Y = [Y; cluster_centres(i,:) + cluster_sd * randn(1, d)]; %#ok<AGROW>
        else
            % Add different noise - potentially structured
%             noise = randn(1, d);
%             noise = noise / sqrt(sum(noise.^2)); % Unit vector
%             noise = noise * (1.5 + 0.05 * randn) * cluster_sd; % Jitter and scale
            noise = cluster_sd * trnd(2,1,d);
            Y = [Y; cluster_centres(i,:) + noise]; %#ok<AGROW>
        end
    end
end

%% Visualise data using PCA

coeff = pca(Y);
Y_pca = Y * coeff(:,1:2);

plot(Y_pca(:,1), Y_pca(:,2), 'o');

%% Fit MoG and sample

n_centres_mog = 5;

options = statset('Display','final','MaxIter',1000);
mog = gmdistribution.fit(Y,n_centres_mog,'Options',options,'Replicates',100);
X = mog.random(size(Y,1));

%% Visualise data using PCA

coeff = pca([X;Y]);
X_pca = X * coeff(:,1:2);
Y_pca = Y * coeff(:,1:2);

h = figure;
plot(X_pca(:,1), X_pca(:,2), 'ro'); hold on;
plot(Y_pca(:,1), Y_pca(:,2), 'go'); hold off;
save2pdf( 'temp/pca.pdf', h, 600, true );

%% MMD test after PCA

alpha = 0.05;
%params.sig = -1;
params.sig = 0.2;
params.shuff = 100;
[~,~,params,p] = mmdTestBoot_jl(X_pca,Y_pca,alpha,params);
display(p);

%% Plot witness function

m = size(X_pca, 1);
n = size(Y_pca, 1);
t = (((fullfact([200,200])-0.5) / 100) - 1) * 3;
K1 = rbf_dot(X_pca, t, params.sig);
K2 = rbf_dot(Y_pca, t, params.sig);
witness = sum(K1, 1)' / m - sum(K2, 1)' / n;
plot3(t(:,1), t(:,2), witness, 'o');
imagesc(reshape(witness(1:end), 200, 200));
colorbar;

%% kPCA

params.sig = 0.2;
K = rbf_dot([X;Y], [X;Y], params.sig);
kPCA_proj = kernelPCA4(K)';

X_kpca = kPCA_proj(1:size(X,1),:);
Y_kpca = kPCA_proj((size(X,1)+1):end,:);

plot(X_kpca(:,1), X_kpca(:,2), 'ro'); hold on;
plot(Y_kpca(:,1), Y_kpca(:,2), 'go'); hold off;

%% FA

[L,psi,T,stats,F] = factoran([X;Y], 2);

X_fa = F(1:size(X,1),:);
Y_fa = F((size(X,1)+1):end,:);

h = figure;
plot(X_fa(:,1), X_fa(:,2), 'ro'); hold on;
plot(Y_fa(:,1), Y_fa(:,2), 'go'); hold off;
save2pdf( 'temp/fa.pdf', h, 600, true );

%% MMD on FA

alpha = 0.05;
params.sig = -1;
%params.sig = 0.2;
params.shuff = 100;
[~,~,params,p] = mmdTestBoot_jl(X_fa,Y_fa,alpha,params);
display(p);

%% Calculate some distances for reference

d1 = sqrt(sq_dist(X_fa', X_fa'));
d2 = sqrt(sq_dist(Y_fa', Y_fa'));
Z_dr = [X_fa;Y_fa];  %aggregate the sample
d3 = sq_dist(Z_dr', Z_dr');
hist([d1(:);d2(:)]);

%% Find an appropriate FA MMD lengthscale by cross validation

% CV for density estimation
folds = 5;
divisions = 50;
distances = sort([d1(:); d2(:)]);
trial_ell = zeros(divisions,1);
for i = 1:(divisions)%-1)
    trial_ell(i) = i * sqrt(0.5) * distances(floor(0.5*numel(distances))) / divisions;
end
m = size(X_fa, 1);
n = size(Y_fa, 1);
d = size(X_fa, 2);
X_perm = X_fa(randperm(m),:);
Y_perm = Y_fa(randperm(n),:);
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

%% MMD on FA

alpha = 0.05;
params.shuff = 100;
[testStat,thresh,params,p] = mmdTestBoot_jl(X_fa,Y_fa,alpha,params);
display(p);

%% Plot witness function

m = size(X_fa, 1);
n = size(Y_fa, 1);
t = (((fullfact([200,200])-0.5) / 100) - 1) * 2.5;
K1 = rbf_dot(X_fa, t, params.sig);
K2 = rbf_dot(Y_fa, t, params.sig);
witness = sum(K1, 1)' / m - sum(K2, 1)' / n;
plot3(t(:,1), t(:,2), witness, 'o');

h = figure;
reshaped = reshape(witness, 200, 200)';
imagesc(reshaped(end:-1:1,:));
colorbar;
save2pdf( 'temp/witness.pdf', h, 600, true );

%% Find peaks of the witness function on fantasies

close all;

ell = params.sig;
x_opt_Y = zeros(size(Y_fa));
witnesses_Y = zeros(size(Y_fa, 1), 1);

for i = 1:size(Y_fa,1)

    x = Y_fa(i,:)';
    witness = rbf_witness(x, X_fa, Y_fa, ell);
    witnesses_Y(i) = witness;
    fprintf('\nwitness=%f\n', witness);

    % Checkgrad
    % options = optimoptions('fminunc','GradObj','on', ...
    %                        'DerivativeCheck', 'on', ...
    %                        'FinDiffType', 'central');
    % fminunc(@(x) rbf_witness(x, X, Y, ell), x, options);  

    if witness >=0 
        % Maximize
        x = minimize_quiet(x, @(x) neg_rbf_witness(x, X_fa, Y_fa, ell), -50);
    else
        x = minimize_quiet(x, @(x) rbf_witness(x, X_fa, Y_fa, ell), -50);
    end
    x_opt_Y(i,:) = x';
    %imagesc(reshape(x, 28, 28)');
    %drawnow;
    
    fprintf('\ni = %d\n', i);
    
end

%% Find peaks of the witness function on test

close all;

ell = params.sig;
x_opt_X = zeros(size(X_fa));
witnesses_X = zeros(size(X_fa, 1), 1);

for i = 1:size(X_fa,1)

    x = X_fa(i,:)';
    witness = rbf_witness(x, X_fa, Y_fa, ell);
    witnesses_X(i) = witness;
    fprintf('\nwitness=%f\n', witness);

%     % Checkgrad
%     options = optimoptions('fminunc','GradObj','on', ...
%                            'DerivativeCheck', 'on', ...
%                            'FinDiffType', 'central');
%     fminunc(@(x) neg_rbf_witness(x, X, Y, ell), x, options);  

    if witness >=0 
        % Maximize
        x = minimize_quiet(x, @(x) neg_rbf_witness(x, X_fa, Y_fa, ell), -50);
    else
        x = minimize_quiet(x, @(x) rbf_witness(x, X_fa, Y_fa, ell), -50);
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

[sorted, idx] = sort(witness_sums, 'descend');