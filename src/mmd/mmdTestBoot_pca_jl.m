%This function implements the MMD two-sample test using a bootstrap
%approach to compute the test threshold.


%Arthur Gretton
%07/12/08

% Modified by James Robert Lloyd, March 2014

%Inputs: 
%        X contains dx columns, m rows. Each row is an i.i.d sample
%        Y contains dy columns, n rows. Each row is an i.i.d sample
%        alpha is the level of the test
%        params.sig is kernel size. If -1, use median distance heuristic.
%        params.shuff is number of bootstrap shuffles used to
%                     estimate null CDF
%        params.bootForce: if this is 1, do bootstrap, otherwise
%                     look for previously saved threshold
%        d is number of dimensions of pca


%Outputs: 
%        thresh: test threshold for level alpha test
%        testStat: test statistic: m * MMD_b (biased)



function [testStat,thresh,params,p] = mmdTestBoot_pca_jl(X,Y,alpha,params,d)

    
m=size(X,1);
n=size(Y,1);

[coeff] = pca([X;Y]);
X_dr = X * coeff(:,1:d);
Y_dr = Y * coeff(:,1:d);

%Set kernel size to median distance between points in aggregate sample
if params.sig == -1
    Z_dr = [X_dr;Y_dr];  %aggregate the sample
    size1=size(Z_dr,1);
    Zmed = Z_dr;
    G = sum((Zmed.*Zmed),2);
    Q = repmat(G,1,size1);
    R = repmat(G',size1,1);
    dists = Q + R - 2*Zmed*(Zmed');
    dists = dists-tril(dists);
    dists=reshape(dists,size1^2,1);
    params.sig = sqrt(0.5*median(dists(dists>0)));  %rbf_dot has factor two in kernel
end

K = rbf_dot(X_dr,X_dr,params.sig);
L = rbf_dot(Y_dr,Y_dr,params.sig);
KL = rbf_dot(X_dr,Y_dr,params.sig);


%MMD statistic. Here we use biased 
%v-statistic.
testStat = (1/m^2) * sum(sum(K)) - (2 / (m * n)) * sum(sum(KL)) + ...
           (1/n^2) * sum(sum(L)); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate bootstrap distribution of test stat
  
MMDarr = [];
Z = [X;Y];
for i=1:params.shuff

    [~,indShuff] = sort(rand(m+n,1)); % Replace with randPerm?
    X_shuff = Z(indShuff(1:m),:);
    Y_shuff = Z(indShuff((m+1):(m+n)),:);
    [coeff] = pca([X_shuff;Y_shuff]);
    X_dr = X_shuff * coeff(:,1:d);
    Y_dr = Y_shuff * coeff(:,1:d);
    K = rbf_dot(X_dr,X_dr,params.sig);
    L = rbf_dot(Y_dr,Y_dr,params.sig);
    KL = rbf_dot(X_dr,Y_dr,params.sig);

    MMDarr = [MMDarr;nan]; %#ok<AGROW>
    MMDarr(i) = (1/m^2) * sum(sum(K)) - (2 / (m * n)) * sum(sum(KL)) + ...
                (1/n^2) * sum(sum(L)); 
            
    display(i);
    p = sum(testStat < MMDarr) / length(MMDarr);
    display(p);
end 

MMDarr = sort(MMDarr);
thresh = MMDarr(round((1-alpha)*params.shuff));
p = sum(testStat < MMDarr) / length(MMDarr);
end
