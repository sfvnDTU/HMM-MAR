function [results, hmm] = grouphmmpermtest(data,T,options,grouping,nperms, hmm_init, verbose)
% Calculates the p-value of a permutation-test with H0 that 
% the KL-divergence between the transition matrices is equal 
% across the two groups specified. Furthermore, it is tested if the initial
% state probabilites over the groups differ. 
% See example of how to use the code in examples/run_grouppermtest.m
%
% INPUT
% data          observations, either celled format or matrix (cf. hmmmar)
% T             length of series
% options       structure with the training options - see documentation
%               NB! Must include the 'grouping' option - only works for two
%               groups atm...
% nperms        number of permutations for null-distribution
% hmm_init      inital hmm-solution without grouping (optional)
% verbose       print time spent for each permutation (optional, default 0)
%
% OUTPUT
% results      struct containing the two fields P (transition matrix) and
%              Pi (initial state probabilities). They have the following
%              subfields
%    pval      the empirical proportion times where the H0 could 
%               not be rejected         
%    KL        symmetrized KL-divergence between the two specified groups
%    KL_null   matrix (size K x nperms) with sym. KL-divergence for each
%               transition row and for each permutation
% hmm           hmm object from group inference
%
% Author: Søren Føns Vind Nielsen, CogSys, DTU (January, 2018)
%          Based on Diego Vidaurre's HMM-MAR toolbox
%
%%%%%%%%%%%%%%%%%%%%
rng('shuffle')
if nargin < 7
    verbose = 0;
end
if nargin < 6 || isempty(hmm_init)
   hmm_init = []; 
   K = options.K;
else
   K = hmm_init.K;
end

% Check for number of groups
ngroups = numel(unique(grouping(:)));
if ngroups~=2
    error(['Test only works for two groups currently.', ...
        '# groups specfified: %i'], ngroups)
end

% Check other inputs 
if nperms < 1000
    warning('You are running a low number of permutations. Do not expect p-value to well estimated...')
end

% Check that grouping variable complies with data
if iscell(data)
    assert( all(size(data) == size(grouping)) & all(size(data) == size(T)) )
else
    assert(length(grouping)==length(T))
end

% Initialize results
kldist_null_trans = nan(K,nperms);
kldist_null_pi = nan(1,nperms);
corr_null_trans = nan(1,nperms);
corr_null_pi = nan(1,nperms);
FO_group = nan(K,ngroups);
FO_group_perm = nan(K,ngroups, nperms);
P_perm = nan(K,K,ngroups,nperms);
Pi_perm = nan(K,ngroups,nperms);

%% HMM Inference

% initialization...
if isempty(hmm_init)
    if iscell(data)
        [hmm_init,Gamma_init,Xi_init]  = hmmmar(data(:),T(:),options); % first run without grouping
    else
        [hmm_init,Gamma_init,Xi_init]  = hmmmar(data,T,options);
    end
else
    if iscell(data)
        [Gamma_init, Xi_init] = hmmdecode(data(:),T(:),hmm_init,0);
    else
        [Gamma_init, Xi_init] = hmmdecode(data,T,hmm_init,0);
    end
end

% remove big-fields
opt_field_names = fieldnames(options);
fields_to_remove = strfind(opt_field_names, 'BIG');
for f = 1:length(opt_field_names)
    if fields_to_remove{f}
        options = rmfield(options, opt_field_names{f});
    end
end

% Convert T to array
if iscell(T)
    T = [T{:}]';
end

% run with groupings
hmm_init.train.grouping = grouping(:);
hmm = hsupdate(Xi_init,Gamma_init,T,hmm_init); % update hmm object to have group transitions

% Calc FO
FO = getFractionalOccupancy(Gamma_init,T);
FO_group(:,1) = mean(FO(grouping==1,:));
FO_group(:,2) = mean(FO(grouping==2,:));

% Calc KL differences
[~, kldist_trans] = KLtransition_dist(hmm.Dir2d_alpha(:,:,1),hmm.Dir2d_alpha(:,:,2)); % rowwise symmetric KL-dist
kldist_pi = KLpi_dist(hmm.Dir_alpha(:,1),hmm.Dir_alpha(:,2));

% Correlation measures
P1 = hmm.P(:,:,1); P2 = hmm.P(:,:,2);
corr_trans = corr(P1(:), P2(:));
corr_pi = corr(hmm.Pi(:,1), hmm.Pi(:,2));
results.P.P = hmm.P;
results.Pi.Pi = hmm.Pi;

%% Post-hoc HMM inference on permuted groups
if nperms > 0
fprintf('Running permutations.... \n')
options.verbose = 0;
for per = 1:nperms
    perm_start_tic = tic;
    perm_grouping = grouping(:,randperm(size(grouping,2)));
    hmm_init.train.grouping = perm_grouping(:);
    hmm_perm = hsupdate(Xi_init,Gamma_init,T,hmm_init); % update hmm object to have group transitions
    
    % Calculate stats on permuted groups
    [~,kldist_null_trans(:,per)] = KLtransition_dist(hmm_perm.Dir2d_alpha(:,:,1),hmm_perm.Dir2d_alpha(:,:,2));
    kldist_null_pi(per) = KLpi_dist(hmm.Dir_alpha(:,1),hmm.Dir_alpha(:,2));
    
    % Fractional occupancy
    FO_group_perm(:,1,per) = mean(FO(hmm_init.train.grouping==1,:));
    FO_group_perm(:,2,per) = mean(FO(hmm_init.train.grouping==2,:));
    
    % Correlation measures
    P1 = hmm_perm.P(:,:,1); P2 = hmm_perm.P(:,:,2);
    corr_null_trans(per) = corr(P1(:), P2(:));
    corr_null_pi(per) = corr(hmm_perm.Pi(:,1), hmm_perm.Pi(:,2));
    
    % Save transtion matrices and Pi vector
    P_perm(:,:,:,per) = hmm_perm.P;
    Pi_perm(:,:,per) = hmm_perm.Pi;
    
    % Timing
    perm_time = toc(perm_start_tic);
    if verbose
       fprintf('Permutation %i took %.4f seconds \n', per, perm_time) 
    end
end

%% Calc p-val pr. state
% Binomial confidence interval (95%) upper bound estimate of p-value

% Transition matrix
kl_gt = bsxfun(@gt, kldist_null_trans, kldist_trans); 
[~,pci] = binofit( sum(kl_gt,2), nperms);
pval_trans = pci(:,2);

[~,pci] = binofit( sum(kldist_null_pi > kldist_pi), nperms);
pval_pi = pci(2);


else
   pval_pi = nan;
   pval_trans = nan;  
end

%% Pack things together in a results struct
% Transition matrix stats
results.P.KL = kldist_trans;
results.P.KL_null = kldist_null_trans;
results.P.KL_pval = pval_trans;
results.P.corr = corr_trans;
results.P.corr_null = corr_null_trans;
results.P.P_perm = P_perm;

% Pi stats
results.Pi.KL = kldist_pi;
results.Pi.KL_null = kldist_null_pi;
results.Pi.KL_pval = pval_pi;
results.Pi.corr = corr_pi;
results.Pi.corr_null = corr_null_pi;
results.Pi.Pi_perm = Pi_perm;

% FO stats
results.FO.Avg = FO_group;
results.FO.Permuted = FO_group_perm;

%eof
end