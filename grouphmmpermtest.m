function [pval, kldist, kldist_null, hmm] = grouphmmpermtest(data,T,options, nperms, hmm_init)
% Calculates the p-value of a permutation-test with H0 that 
% the KL-divergence between the transition matrices is equal 
% across the two groups specified
% See example of how to use the code in examples/run_grouppermtest.m
%
%  Dependencies: 
%            Durga Lal Shrestha's invprctile script from MATLAB-central
% https://se.mathworks.com/matlabcentral/fileexchange/41131-inverse-percentiles-of-a-sample
%
% INPUT
% data          observations, either celled format or matrix (cf. hmmmar)
% T             length of series
% options       structure with the training options - see documentation
%               NB! Must include the 'grouping' option - only works for two
%               groups atm...
% nperms        number of permutations for null-distribution
% hmm_init      inital hmm-solution without grouping (optional)
%
% OUTPUT
% ??           < ?? A GULL ?? >         
% ??           < ?? A GULL ?? >
%
% Author: Søren Føns Vind Nielsen, CogSys, DTU (January, 2018)
%          Based on Diego Vidaurre's HMM-MAR toolbox
%
%%%%%%%%%%%%%%%%%%%%
if nargin<5
   hmm_init = []; 
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
kldist_null = nan(1,nperms);

%% HMM Inference

% initialization...
grouping = options.grouping;
options = rmfield(options, 'grouping');
if isempty(hmm_init)
    [hmm_init,Gamma_init,Xi_init]  = hmmmar(data,T,options); % first run without grouping
else
    [Gamma_init, Xi_init] = hmmdecode(data,T,hmm_init,0);
end

% run with groupings
hmm_init.train.grouping = grouping;
hmm = hsupdate(Xi_init,Gamma_init,T,hmm_init); % update hmm object to have group transitions
[Gamma,~,Xi] = hsinference(data,T,hmm); % estimate relevant expectations
hmm = hsupdate(Xi,Gamma,T,hmm); % update final transition matrices
kldist = KLtransition_dist(hmm.Dir2d_alpha(:,:,1),hmm.Dir2d_alpha(:,:,2));


%% Post-hoc HMM inference on permuted groups
fprintf('Running permutations.... \n')
options.verbose = 0;
tic
for per = 1:nperms
    options.grouping = grouping(randperm(length(grouping)));
    hmm_init.train.grouping = options.grouping;
    hmm_perm = hsupdate(Xi_init,Gamma_init,T,hmm_init); % update hmm object to have group transitions
    [Gamma,~,Xi] = hsinference(data,T,hmm_perm); % estimate relevant expectations
    hmm_perm = hsupdate(Xi,Gamma,T,hmm_perm); % update final transition matrices
    kldist_null(per) = KLtransition_dist(hmm_perm.Dir2d_alpha(:,:,1),hmm_perm.Dir2d_alpha(:,:,2));
end
toc

%% Calc p-val
invp = invprctile(kldist_null, kldist);
pval = (100-invp)/100;

%eof
end