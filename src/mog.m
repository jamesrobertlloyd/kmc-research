%% Clear and load mmd tools

clear all;
addpath(genpath('./mmd'));

%% Let's try MoG

n_per_cluster = 500;
dof = 3;
c = [repmat([1,0,0], n_per_cluster, 1);
     repmat([0,1,0], n_per_cluster, 1);
     repmat([0,0,1], n_per_cluster, 1)];
m = 0.5*[1, 1;
     -1, 1;
     0, -1];
y_c = c * m;
y1 = y_c + 0.1*randn(size(y_c));
m = 0.5*[1, 1;
     -1, 1;
     0, -1];
y_c = c * m;
y2 = y_c;
y2(1:(2*n_per_cluster),:) = y2(1:(2*n_per_cluster),:) + 0.1*randn(size(y2(1:(2*n_per_cluster),:)));
y2((2*n_per_cluster+1):end,:) = y2((2*n_per_cluster+1):end,:) + 0.1*trnd(dof,size(y2((2*n_per_cluster+1):end,:)));

alpha = 0.10;
params.sig = -1;
params.sig = 0.3;
params.shuff = 100;
[testStat,thresh,params] = mmdTestBoot_jl(y1,y2,alpha,params);
testStat
thresh
params

testStat / thresh

m = size(y1, 1);
n = size(y1, 1);
t = (((fullfact([50,50])-0.5) / 25) - 1) * 1;
K1 = rbf_dot(y1, t, params.sig);
K2 = rbf_dot(y2, t, params.sig);
witness = sum(K1, 1)' / m - sum(K2, 1)' / n;
plot3(t(:,1), t(:,2), witness, 'o');
imagesc(reshape(witness(end:-1:1), 50, 50)');
colorbar;

%% Let's try MoG with a circle

n_per_cluster = 500;
dof = 3;
c = [repmat([1,0,0], n_per_cluster, 1);
     repmat([0,1,0], n_per_cluster, 1);
     repmat([0,0,1], n_per_cluster, 1)];
m = 0.5*[1, 1;
     -1, 1;
     0, -1];
y_c = c * m;
y1 = y_c + 0.1*randn(size(y_c));
m = 0.5*[1, 1;
     -1, 1;
     0, -1];
y_c = c * m;
y2 = y_c;
y2(1:(2*n_per_cluster),:) = y2(1:(2*n_per_cluster),:) + 0.1*randn(size(y2(1:(2*n_per_cluster),:)));
rand_ang = 2*pi*rand(n_per_cluster,1);
rand_loc = [cos(rand_ang), sin(rand_ang)];
rand_loc = rand_loc .* (0.1+0.1*repmat(rand(n_per_cluster,1),1,2));
y2((2*n_per_cluster+1):end,:) = y2((2*n_per_cluster+1):end,:) + rand_loc;

alpha = 0.05;
%params.sig = -1;
params.sig = 0.05;
params.shuff = 100;
[testStat,thresh,params] = mmdTestBoot_jl(y1,y2,alpha,params);
testStat
thresh
params

testStat / thresh

m = size(y1, 1);
n = size(y1, 1);
t = (((fullfact([50,50])-0.5) / 25) - 1) * 1;
K1 = rbf_dot(y1, t, params.sig);
K2 = rbf_dot(y2, t, params.sig);
witness = sum(K1, 1)' / m - sum(K2, 1)' / n;
plot3(t(:,1), t(:,2), witness, 'o');
imagesc(reshape(witness(end:-1:1), 50, 50)');
colorbar;

sorted_witness = sort(witness);
plot(y2(:,1), y2(:,2), 'go');
xlim([-1,1]);
ylim([-1,1]);
hold on;
plot(t(witness>=sorted_witness(end-20),1), t(witness>=sorted_witness(end-20),2), 'rx');
plot(t(witness<=sorted_witness(20),1), t(witness<=sorted_witness(20),2), 'b+');
hold off;

%% Calculate some distances

d1 = sqrt(sq_dist(y1', y1'));
d2 = sqrt(sq_dist(y2', y2'));
hist([d1(:);d2(:)]);