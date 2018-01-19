function [results, hmm] = grouphmmpermtest(data,T,options, nperms, hmm_init, verbose)
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
if nargin < 6
    verbose = 0;
end
if nargin<5 || isempty(hmm_init)
   hmm_init = []; 
   K = options.K;
else
   K = hmm_init.K;
end

% Check if grouping is specified
if ~isfield(options, 'grouping')
    error('Please specify the grouping field in options...')
end

% Check for number of groups
ngroups = numel(unique(options.grouping));
if ngroups~=2
    error(['Test only works for two groups currently.', ...
        '# groups specfified: %i'], ngroups)
end

% Check other inputs 
if nperms < 1000
    warning('You are running a low number of permutations. Do not expect p-value to well estimated...')
end
% TODO: Add checking of input data...

% Initialize results
kldist_null_trans = nan(K,nperms);
kldist_null_pi = nan(1,nperms);
corr_null_trans = nan(1,nperms);
corr_null_pi = nan(1,nperms);
FO_group = nan(K,ngroups);
FO_group_perm = nan(K,ngroups, nperms);

%% HMM Inference

% initialization...
grouping = options.grouping;
options = rmfield(options, 'grouping');
if isempty(hmm_init)
    [hmm_init,Gamma_init,Xi_init]  = hmmmar(data,T,options); % first run without grouping
else
    [Gamma_init, Xi_init] = hmmdecode(data,T,hmm_init,0);
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
hmm_init.train.grouping = grouping;
hmm = hsupdate(Xi_init,Gamma_init,T,hmm_init); % update hmm object to have group transitions
[Gamma,~,Xi] = hsinference(data,T,hmm); % estimate relevant expectations
FO = getFractionalOccupancy(Gamma,T);
FO_group(:,1) = mean(FO(grouping==1,:));
FO_group(:,2) = mean(FO(grouping==2,:));
hmm = hsupdate(Xi,Gamma,T,hmm); % update final transition matrices
[~, kldist_trans] = KLtransition_dist(hmm.Dir2d_alpha(:,:,1),hmm.Dir2d_alpha(:,:,2)); % rowwise symmetric KL-dist
kldist_pi = KLpi_dist(hmm.Dir_alpha(:,1),hmm.Dir_alpha(:,2));

% Correlation measures
P1 = hmm.P(:,:,1); P2 = hmm.P(:,:,2);
corr_trans = corr(P1(:), P2(:));
corr_pi = corr(hmm.Pi(:,1), hmm.Pi(:,2));


%% Post-hoc HMM inference on permuted groups
fprintf('Running permutations.... \n')
options.verbose = 0;
for per = 1:nperms
    perm_start_tic = tic;
    options.grouping = grouping(randperm(length(grouping)));
    hmm_init.train.grouping = options.grouping;
    hmm_perm = hsupdate(Xi_init,Gamma_init,T,hmm_init); % update hmm object to have group transitions
    [Gamma,~,Xi] = hsinference(data,T,hmm_perm); % estimate relevant expectations
    hmm_perm = hsupdate(Xi,Gamma,T,hmm_perm); % update final transition matrices
    [~,kldist_null_trans(:,per)] = KLtransition_dist(hmm_perm.Dir2d_alpha(:,:,1),hmm_perm.Dir2d_alpha(:,:,2));
    kldist_null_pi(per) = KLpi_dist(hmm.Dir_alpha(:,1),hmm.Dir_alpha(:,2));
    
    % Fractional occupancy
    FO = getFractionalOccupancy(Gamma,T);
    FO_group_perm(:,1,per) = mean(FO(options.grouping==1,:));
    FO_group_perm(:,2,per) = mean(FO(options.grouping==2,:));
    
    % Correlation measures
    P1 = hmm_perm.P(:,:,1); P2 = hmm_perm.P(:,:,2);
    corr_null_trans(per) = corr(P1(:), P2(:));
    corr_null_pi(per) = corr(hmm_perm.Pi(:,1), hmm_perm.Pi(:,2));
    
    % Timing
    perm_time = toc(perm_start_tic);
    if verbose
       fprintf('Permutation %i took %.4f seconds \n', per, perm_time) 
    end
end

%% Calc p-val pr. state
% Fraction of times null is smaller than actual kldist (pr. row)
kl_lt = bsxfun(@lt, kldist_null_trans, kldist_trans); 
pval_trans = 1 - mean(kl_lt,2);

pval_pi = 1 - mean(kldist_null_pi < kldist_pi);

%% Pack things together in a results struct
% Transition matrix stats
results.P.KL = kldist_trans;
results.P.KL_null = kldist_null_trans;
results.P.KL_pval = pval_trans;
results.P.corr = corr_trans;
results.P.corr_null = corr_null_trans;

% Pi stats
results.Pi.KL = kldist_pi;
results.Pi.KL_null = kldist_null_pi;
results.Pi.KL_pval = pval_pi;
results.Pi.corr = corr_pi;
results.Pi.corr_null = corr_null_pi;

% FO stats
results.FO.Avg = FO_group;
results.FO.Permuted = FO_group_perm;

%eof
end