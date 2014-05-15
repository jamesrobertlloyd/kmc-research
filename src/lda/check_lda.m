%% Latent Dirichlet Allocation applied to the KOS dataset

load kos_doc_data.mat
addpath(genpath('../mmd'));
addpath(genpath('../util'));

%% Run LDA

W = max([A(:,2); B(:,2)]);  % number of unique words
D = max(A(:,1));            % number of documents in A
K = 20;                     % number of mixture components we will use

alpha = 0.1;    % parameter of the Dirichlet over topics for one document
gamma = 0.1;    % parameter of the Dirichlet over words

% A's columns are doc_id, word_id, count
swd = sparse(A(:,2),A(:,1),A(:,3));
Swd = sparse(B(:,2),B(:,1),B(:,3));

% Initialization: assign each word in each document a topic
skd = zeros(K,D); % count of word assignments to topics for document d
swk = zeros(W,K); % unique word topic assignment counts accross all documents
s = cell(D, 1);   % one cell for every document
for d = 1:D                % cycle through the documents
  z = zeros(W,K);          % unique word topic assignment counts for doc d
  for w = A(A(:,1)==d,2)'  % loop over unique words present in document d
    c = swd(w,d);          % number of occurences of word w in document d
    for i=1:c    % assign each occurence of word w to a topic at random
      k = ceil(K*rand());
      z(w,k) = z(w,k) + 1;
    end
  end
  skd(:,d) = sum(z,1)';  % number of words in doc d assigned to each topic
  swk = swk + z;  % unique word topic assignment counts accross all documents
  s{d} = sparse(z); % sparse representation: z contains many zero entries
end
sk = sum(skd,2);  % word to topic assignment counts accross all documents

Gibbs_Samples = 10;

posterior_entropies = zeros (K, Gibbs_Samples);
posterior_topics = alpha * ones (K, Gibbs_Samples);

% This makes a number of Gibbs sampling sweeps through all docs and words
for iter = 1:Gibbs_Samples     % This can take a couple of minutes to run
  display (iter);
  for d = 1:D
    z = full(s{d}); % unique word topic assigmnet counts for document d
    for w = A(A(:,1)==d,2)' % loop over unique words present in document d
      a = z(w,:); % number of times word w is assigned to each topic in doc d
      ka = find(a); % topics with non-zero counts for word d in document d
      for k = ka(randperm(length(ka))) % loop over topics in permuted order
        for i = 1:a(k) % loop over counts for topic k
          z(w,k) = z(w,k) - 1;      % remove word from count matrices
          swk(w,k) = swk(w,k) - 1;
          sk(k)    = sk(k)    - 1;
          skd(k,d) = skd(k,d) - 1;
          b = (alpha + skd(:,d)) .* (gamma + swk(w,:)') ./ (W*gamma + sk);
          kk = sampDiscrete(b);     % Gibbs sample new topic assignment
          z(w,kk) = z(w,kk) + 1;    % add word with new topic to count matrices
          skd(kk,d) = skd(kk,d) + 1;
          swk(w,kk) = swk(w,kk) + 1;
          sk(kk) =    sk(kk)    + 1;
        end
      end
    end
    s{d} = sparse(z);   % store back into sparse structure
    %Calculate posterior
    posterior_topics(:, iter) = posterior_topics(:, iter) + ...
                                skd(:,d) / sum(skd(:,d));
  end
  %Calculate entropy for each topic
  posterior_params = swk + gamma * ones(size(swk));
  posterior_probs = posterior_params ./ ...
                    repmat (sum(posterior_params, 1), W, 1);
  posterior_entropies(:, iter) = -sum (posterior_probs .* log2 (posterior_probs))';
  %Calculate posterior
  %posterior_topics(:, iter) = posterior_topics(:, iter) + sk;
end

%% PCA visualisation of the original data

fdw = full(swd)';
fdw_probs = fdw ./ repmat(sum(fdw, 2), 1, size(fdw, 2));

d = 2;
[coeff,~,latent] = pca(fdw_probs);
fdw_probs_dr = fdw_probs * coeff(:,1:d);

plot(fdw_probs_dr(:,1), fdw_probs_dr(:,2), 'go');

%% Generate some data from the posterior - same shape as original

%%%% Posterior probs is not actually the posterior - to be updated

n_words = full(sum(swd, 1))';

posterior_docs = zeros(D, W);

topic_word_prob = posterior_probs';

for i = 1:D
    % Sample topic distribution
    theta = dirichlet_sample(repmat(alpha, 20, 1));
    word_prob = theta' * topic_word_prob;
    for j = 1:n_words(i)
        w = sampDiscrete(word_prob);
        posterior_docs(i, w) = posterior_docs(i, w) + 1;
    end
    if mod(i, 100) == 0
        display(i);
    end
end

%% PCA on generated data

post_docs_probs = posterior_docs ./ repmat(sum(posterior_docs, 2), 1, size(posterior_docs, 2));

d = 2;
[coeff,~,latent] = pca(post_docs_probs);
post_docs_probs_dr = post_docs_probs * coeff(:,1:d);

plot(post_docs_probs_dr(:,1), post_docs_probs_dr(:,2), 'ro');

%% Joint PCA - consider a random projection first for speed

X = fdw_probs;
Y = post_docs_probs;

% Random projection

rand_proj = randn(W, 1000);
X = X * rand_proj;
Y = Y * rand_proj;

% PCA

d = 2;
[coeff,~,latent] = pca([X;Y]);
X_dr = X * coeff(:,1:d);
Y_dr = Y * coeff(:,1:d);

plot(X_dr(:,1), X_dr(:,2), 'go'); hold on;
plot(Y_dr(:,1), Y_dr(:,2), 'ro'); hold off;

%% Calculate some distances for reference

d1 = sqrt(sq_dist(X_dr', X_dr'));
d2 = sqrt(sq_dist(Y_dr', Y_dr'));
Z_dr = [X_dr;X_dr];  %aggregate the sample
d3 = sq_dist(Z_dr', Z_dr');
figure;
hist([d1(:);d2(:)]);
    
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
m = size(X_dr, 1);
n = size(Y_dr, 1);
d = size(X_dr, 2);
X_perm = X_dr(randperm(m),:);
Y_perm = Y_dr(randperm(n),:);
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

%% MMD on PCA reduced variables

alpha = 0.05;
params.shuff = 100;
[testStat,thresh,params,p] = mmdTestBoot_jl(X_dr,Y_dr,alpha,params);
display(p);
%pause;

%% Compute witness function in 2d

if size(X_dr,2) == 2
    m = size(X_dr, 1);
    n = size(Y_dr, 1);
    t = (((fullfact([200,200])-0.5) / 200) - 0) * 1;
    t = t .* (1.4 * repmat(range([X_dr; Y_dr]), size(t,1), 1));
    t = t + repmat(min([X_dr; Y_dr]) - 0.2*range([X_dr; Y_dr]), size(t,1), 1);
    K1 = rbf_dot(X_dr, t, params.sig);
    K2 = rbf_dot(Y_dr, t, params.sig);
    witness = sum(K1, 1)' / m - sum(K2, 1)' / n;
    %plot3(t(:,1), t(:,2), witness, 'bo');
    %hold on;
    %plot3(Y_dr(:,1), Y_dr(:,2), repmat(max(max(witness)), size(Y_dr)), 'ro');
    reshaped = reshape(witness, 200, 200)';

    h = figure;
    imagesc(reshaped(end:-1:1,:));
    colorbar;
    save2pdf('../temp/witness.pdf', h, 600, true );
    %hold off;
end