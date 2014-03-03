%% Clear and load mmd tools

clear all;
close all;
addpath(genpath('./mmd'));
addpath(genpath('./util'));

%% Init PRNGs

seed = 1;
init_rand(seed);

%% Create some high dimensional mixture of Gaussians

n_per_cluster = 500;
d = 2;
effective_d = 2;
n_clusters = 3;
cluster_sd = 0.2;

basis = randn(d, effective_d);

cluster_centres = (basis * randn(effective_d, n_clusters))';

Y = [];
for i = 1:n_clusters
    for j = 1:n_per_cluster
        if i < n_clusters
            % Add Gaussian noise
            Y = [Y; cluster_centres(i,:) + cluster_sd * randn(1, d)]; %#ok<AGROW>
        else
            % Add different noise - potentially structured
            noise = randn(1, d);
            noise = noise / sqrt(sum(noise.^2)); % Unit vector
            noise = noise * (1.5 + 0.05 * randn) * cluster_sd; % Jitter and scale
            Y = [Y; cluster_centres(i,:) + noise]; %#ok<AGROW>
        end
    end
end

%% Visualise data using PCA

coeff = pca(Y);
Y_pca = Y * coeff(:,1:2);

plot(Y_pca(:,1), Y_pca(:,2), 'o');

%% Fit MoG and sample

n_centres_mog = 3;

options = statset('Display','final','MaxIter',1000);
mog = gmdistribution.fit(Y,n_centres_mog,'Options',options,'Replicates',10);
X = mog.random(size(Y,1));

%% Visualise data using PCA

coeff = pca([X;Y]);
X_pca = X * coeff(:,1:2);
Y_pca = Y * coeff(:,1:2);

plot(X_pca(:,1), X_pca(:,2), 'ro'); hold on;
plot(Y_pca(:,1), Y_pca(:,2), 'go'); hold off;

%% MMD test after PCA

alpha = 0.05;
%params.sig = -1;
params.sig = 0.2;
params.shuff = 100;
[testStat,thresh,params,p] = mmdTestBoot_jl(X_pca,Y_pca,alpha,params);
display(p);

%% Plot witness function

m = size(X_pca, 1);
n = size(Y_pca, 1);
t = (((fullfact([200,200])-0.5) / 100) - 1) * 3;
K1 = rbf_dot(X_pca, t, params.sig);
K2 = rbf_dot(Y_pca, t, params.sig);
witness = sum(K1, 1)' / m - sum(K2, 1)' / n;
plot3(t(:,1), t(:,2), witness, 'o');
%imagesc(reshape(witness(1:end), 200, 200)');
%colorbar;