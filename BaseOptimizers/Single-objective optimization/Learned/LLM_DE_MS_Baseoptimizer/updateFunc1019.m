% MATLAB Code
function [offspring] = updateFunc1019(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints and fitness
    sigma_c = std(cons) + 1e-12;
    sigma_f = std(popfits) + 1e-12;
    norm_cons = (cons - mean(cons)) / sigma_c;
    norm_fits = (popfits - mean(popfits)) / sigma_f;
    
    % Combined ranking (lower is better)
    alpha = 0.7; % fitness weight
    beta = 0.3;  % constraint weight
    ranks = tiedrank(alpha*norm_fits - beta*norm_cons);
    
    % Elite selection (top 30%)
    elite_mask = ranks <= 0.3*NP;
    elite_weights = 1./ranks(elite_mask);
    elite_weights = elite_weights / sum(elite_weights);
    x_elite = sum(popdecs(elite_mask,:) .* elite_weights(:), 1);
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) randi(NP-1), idx);
    r1 = r1 + (r1 >= idx);
    r2 = arrayfun(@(i) randi(NP-1), idx);
    r2 = r2 + (r2 >= idx);
    
    % Adaptive parameters
    max_cons = max(abs(cons)) + 1e-12;
    F = 0.5 + 0.3 * tanh(2 * norm_cons); % Scaling factor
    eta = 0.2 * (1 - abs(cons)/max_cons) .* randn(NP, 1); % Noise term
    CR = 0.85 - 0.35 * (ranks/NP); % Crossover rate
    
    % Direction vectors
    dir_sign = sign(popfits - mean(popfits));
    dir_vectors = (x_elite - popdecs) .* dir_sign(:);
    
    % Mutation
    diff_vectors = popdecs(r1,:) - popdecs(r2,:);
    mutants = popdecs + F(:).*dir_vectors + eta(:).*diff_vectors;
    
    % Crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with bounce-back
    lb_viol = offspring < lb;
    ub_viol = offspring > ub;
    offspring(lb_viol) = lb(lb_viol) + 0.5*rand(sum(lb_viol(:)),1).*(popdecs(lb_viol)-lb(lb_viol));
    offspring(ub_viol) = ub(ub_viol) - 0.5*rand(sum(ub_viol(:)),1).*(ub(ub_viol)-popdecs(ub_viol));
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub), lb);
end