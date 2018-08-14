function [KLdiv, KLdiv_rowwise] = KLdisc_dist(A1,A2)
% Compute symmetric KL-divergence between two discrete distributions
% specified in vectors A1 and A2
% If A1 and A2 are matrices the divergence is calculated rowwise and summed
assert(all(size(A1)==size(A2)))

K = size(A1,1); % number of states
KLdiv_rowwise = nan(K,1);
for k = 1:K
    kl12 = sum(A1(k,:).*(log(A1(k,:)) - log(A2(k,:))));
    kl21 = sum(A2(k,:).*(log(A2(k,:)) - log(A1(k,:))));
    KLdiv_rowwise(k) = (kl12+kl21)/2;
end
KLdiv = sum(KLdiv_rowwise);
%eof
end