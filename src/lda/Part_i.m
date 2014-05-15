% Latent Dirichlet Allocation applied to the KOS dataset

% ADVICE: consider doing clear, close all

clear all
close all
load kos_doc_data.mat

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

%Plot evolution of posterior

[sorted, sort_indices] = sort (posterior_topics(:, end)', 'descend');

h = figure;
hold on;
for i = 1:10
  c = (10 - i) / 10;
  plot (1:K, posterior_topics(sort_indices, i) / sum(posterior_topics(:, i)), '+', 'Color', [c c c]);
end
xlabel ({'Topics'}, 'FontSize', 15, 'Interpreter', 'latex');
ylabel ({'Posterior probability'}, 'FontSize', 15, 'Interpreter', 'latex');
title  ({'Posterior probabilities of topics - sorted'}, 'FontSize', 15,...
         'Interpreter', 'latex');
hold off;
save2pdf('Part_i_topics.pdf', h, 150);

%Plot evolution of entropies

h = figure;
hold on;
for i = 1:10
  c = (10 - i) / 10;
  plot (1:K, posterior_entropies(sort_indices, i), '+', 'Color', [c c c]);
end
xlabel ({'Topics'}, 'FontSize', 15, 'Interpreter', 'latex');
ylabel ({'Posterior entropy'}, 'FontSize', 15, 'Interpreter', 'latex');
title  ({'Posterior entropies of topic models'}, 'FontSize', 15,...
         'Interpreter', 'latex');
hold off;
save2pdf('Part_i_entropies.pdf', h, 150);

% compute the perplexity for all words in the test set B
% We need the new Skd matrix, derived from corpus B
lp = 0; nd = 0;
for d = unique(B(:,1))'  % loop over all documents in B
  display (max (unique(B(:,1))) - d)
  % randomly assign topics to each word in test document d
  z = zeros(W,K);
  for w = B(B(:,1)==d,2)'   % w are the words in doc d
    for i=1:Swd(w,d)
      k = ceil(K*rand());
      z(w,k) = z(w,k) + 1;
    end
  end
  Skd = sum(z,1)';
  Sk = sk + Skd;  
  % perform some iterations of Gibbs sampling for test document d
  for iter = 1:10
    for w = B(B(:,1)==d,2)' % w are the words in doc d
      a = z(w,:); % number of times word w is assigned to each topic in doc d
      ka = find(a); % topics with non-zero counts for word d in document d
      for k = ka(randperm(length(ka)))
        for i = 1:a(k)
          z(w,k) = z(w,k) - 1;   % remove word from count matrix for doc d
          Skd(k) = Skd(k) - 1;
          b = (alpha + Skd) .* (gamma + swk(w,:)') ./ (W*gamma + sk);
          kk = sampDiscrete(b);
          z(w,kk) = z(w,kk) + 1; % add word with new topic to count matrix for doc d
          Skd(kk) = Skd(kk) + 1;
        end
      end
    end
  end
  b=(alpha+Skd')/sum(alpha+Skd)*bsxfun(@rdivide,gamma+swk',W*gamma+sk);  
  w=B(B(:,1)==d,2:3);
  lp = lp + log(b(w(:,1)))*w(:,2);   % log probability, doc d
  nd = nd + sum(w(:,2));             % number of words, doc d
end
perplexity = exp(-lp/nd)   % perplexity

%%

% this code allows looking at top I words for each mixture component
I = 5;
for k=sort_indices, [i ii] = sort(-swk(:,k)); ZZ(k,1:I)=ii(1:I); end
for i=1:I, for k=sort_indices, fprintf('%-15s',V{ZZ(k,i)}); end; fprintf('\n'); end

%% Some extra analysis

[sorted, sort_indices] = sort (posterior_topics(:, end)', 'descend');
topic_probs = posterior_topics(sort_indices, end) / sum(posterior_topics(:, end));

topic_probs = skd ./ repmat(sum(skd, 1), K, 1);

%% PCA visualisation of the original data

fdw = full(swd)';
fdw_probs = fdw ./ repmat(sum(fdw, 2), 1, size(fdw, 2));

d = 2;
[coeff,~,latent] = pca(fdw_probs);
fdw_probs_dr = fdw_probs * coeff(:,1:d);

plot(fdw_probs_dr(:,1), fdw_probs_dr(:,2), 'go');

%% Generate some data from the posterior

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

%% PCA

post_docs_probs = posterior_docs ./ repmat(sum(posterior_docs, 2), 1, size(posterior_docs, 2));

d = 2;
[coeff,~,latent] = pca(post_docs_probs);
post_docs_probs_dr = post_docs_probs * coeff(:,1:d);

plot(post_docs_probs_dr(:,1), post_docs_probs_dr(:,2), 'ro');

%% Joint PCA - consider a random projection first for speed

d = 2;
[coeff,~,latent] = pca([fdw_probs;post_docs_probs]);
fdw_probs_dr = fdw_probs * coeff(:,1:d);
post_docs_probs_dr = post_docs_probs * coeff(:,1:d);

plot(fdw_probs_dr(:,1), fdw_probs_dr(:,2), 'go'); hold on;
plot(post_docs_probs_dr(:,1), post_docs_probs_dr(:,2), 'ro'); hold off;