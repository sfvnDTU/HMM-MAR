%% Run Group HMM Permutation Test
% This script exemplifies how to use the Group-HMM permutation test
% framework (cf. grouphmmpermtest.m)
% The two groups are assumed to share the same state parameters (eg.
% covariance matrices and mean-values), whereas the transition
% probabilities between states varies over the groups. This script
% generates some data with two groups that complies with the above
% assumption and plots the null-distribution of the symmetrized KL-divergence
% between each row of the transition matrices. 

% Parameters
K = 3; % number of states in the model
ndim = 4;
Nsubs = 10; % number of subjects in each group
T = 500*ones(1, Nsubs); % T for each of the two groups
nperms = 10000;

%% Simulate data

% base hmm object
hmmtrue = struct();
hmmtrue.K = K;
hmmtrue.state = struct();
hmmtrue.train.covtype = 'full';
hmmtrue.train.zeromean = 0;
hmmtrue.train.order = 0;
for k = 1:K
    hmmtrue.state(k).W.Mu_W = 3*rand(1,ndim);
    R = triu(randn(ndim));
    hmmtrue.state(k).Omega.Gam_rate = R'*R;
    hmmtrue.state(k).Omega.Gam_shape = 1000;
end
hmmtrue.Pi = ones(1,K);
hmmtrue.Pi = hmmtrue.Pi./sum(hmmtrue.Pi);

% Generate the two different transition matrices
transtrue = nan(K,K,2);
diagonal = 5;
% Group 1
trans_tmp = diagonal*eye(K)+ 1e-2*diagonal*ones(K);
trans_tmp(1,2) = diagonal*0.5;
transtrue(:,:,1) = bsxfun(@times,trans_tmp, 1./sum(trans_tmp,2) );
% Group 2
trans_tmp = diagonal*eye(K) + 1e-2*diagonal*ones(K);
transtrue(:,:,2) = bsxfun(@times,trans_tmp, 1./sum(trans_tmp,2) );

% Simulate data
% Group 1
hmm_1 = hmmtrue;
hmm_1.P = transtrue(:,:,1);
X1 = simhmmmar(T,hmm_1,[]);

% Group 2
hmm_2 = hmmtrue;
hmm_2.P = transtrue(:,:,2);
X2 = simhmmmar(T,hmm_2,[]);

groups = [ones(1,Nsubs),2*ones(1,Nsubs)]; % group assigments
X = [X1;X2];


%% Run test

% hmm options
options = struct();
options.K = K;
options.covtype = 'full';
options.zeromean = 0;
options.standardise = 0;
options.grouping = groups;
options.order = 0;

[pval, kldist, kldist_null, hmm] = grouphmmpermtest(X,[T,T],options,nperms);

% Plot null distribution
figure,
for k = 1:hmm.K
    subplot(1,hmm.K,k)
    hist(kldist_null(k,:)), hold on
    line([kldist(k), kldist(k)], get(gca,'YLim'),'Color',[1 0 0])
    hold off
    title(sprintf('state %i, p: %.3f',k,pval(k)))
    xlabel('Sym. KL-div')
end

% Plot estimated transition matrix
figure,
for i = 1:size(hmm.P,3)
    subplot(1,size(hmm.P,3),i)
    imagesc(hmm.P(:,:,i), [0,1])
    axis image
    title(sprintf('Group %i', i))
    colorbar
end
