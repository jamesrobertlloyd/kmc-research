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
xlim([-8, 6]);
ylim([-4, 6]);
save2pdf( 'temp/pca.pdf', h, 600, true );

%% MMD test after PCA

alpha = 0.05;
params.sig = -1;
%params.sig = 0.2;
params.shuff = 1000;
[~,~,params,p] = mmdTestBoot_jl(X_pca,Y_pca,alpha,params);
display(p);

%% MMD test with PCA in loop

alpha = 0.05;
params.shuff = 100;
d = 2;
[~,~,params,p] = mmdTestBoot_pca_jl(X,Y,alpha,params,d);
display(p);

%% Plot witness function

m = size(X_pca, 1);
n = size(Y_pca, 1);
t = fullfact([200,200]) / 200;
t(:,1) = t(:,1) * (6 + 8) - 8;
t(:,2) = t(:,2) * (6 + 4) - 4;
K1 = rbf_dot(X_pca, t, params.sig);
K2 = rbf_dot(Y_pca, t, params.sig);
witness = sum(K1, 1)' / m - sum(K2, 1)' / n;
plot3(t(:,1), t(:,2), witness, 'o');
imagesc(reshape(witness(1:end), 200, 200));
colorbar;

%% Calculate some distances for reference

d1 = sqrt(sq_dist(X_pca', X_pca'));
d2 = sqrt(sq_dist(Y_pca', Y_pca'));
Z_dr = [X_pca;Y_pca];  %aggregate the sample
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
m = size(X_pca, 1);
n = size(Y_pca, 1);
d = size(X_pca, 2);
X_perm = X_pca(randperm(m),:);
Y_perm = Y_pca(randperm(n),:);
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

%% MMD test after PCA

alpha = 0.05;
params.shuff = 1000;
[~,~,params,p] = mmdTestBoot_jl(X_pca,Y_pca,alpha,params);
display(p);

%% MMD test with PCA in loop

alpha = 0.05;
params.shuff = 100;
d = 2;
[~,~,params,p] = mmdTestBoot_pca_jl(X,Y,alpha,params,d);
display(p);

%% Plot witness function

m = size(X_pca, 1);
n = size(Y_pca, 1);
t = fullfact([200,200]) / 200;
t(:,1) = t(:,1) * (6 + 8) - 8;
t(:,2) = t(:,2) * (6 + 4) - 4;
K1 = rbf_dot(X_pca, t, params.sig);
K2 = rbf_dot(Y_pca, t, params.sig);
witness = sum(K1, 1)' / m - sum(K2, 1)' / n;
plot3(t(:,1), t(:,2), witness, 'o');

h = figure;
reshaped = reshape(witness, 200, 200)';
imagesc(reshaped(end:-1:1,:));
colorbar;
set(gca,'xticklabel',{[]}) ;
set(gca,'yticklabel',{[]}) ;
save2pdf( 'temp/witness.pdf', h, 600, true );