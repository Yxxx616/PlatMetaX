% MATLAB Code
function [offspring] = updateFunc1054(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Elite direction with exponential fitness weights
    sigma_f = std(popfits) + eps;
    weights = exp(-popfits/sigma_f);
    elite_vec = (weights' * popdecs) / sum(weights);
    
    % 2. Constraint-aware parameters
    max_cons = max(abs(cons)) + eps;
    alpha = 1 - abs(cons)/max_cons;
    beta = 0.5 * (1 + alpha);
    
    % 3. Random indices for differential vectors
    idx = randperm(NP);
    r1 = idx(1:NP);
    r2 = idx(mod(1:NP, NP)+1);
    r3 = idx(mod(2:NP+1, NP)+1);
    r4 = idx(mod(3:NP+2, NP)+1);
    
    % 4. Hybrid mutation with adaptive scaling
    F1 = 0.6 + 0.2 * rand(NP,1);
    F2 = 0.4 + 0.2 * rand(NP,1);
    F3 = 0.2 + 0.1 * rand(NP,1);
    
    elite_term = F1.*beta .* (elite_vec - popdecs);
    diff_term1 = F2.*(1-alpha) .* (popdecs(r1,:) - popdecs(r2,:));
    diff_term2 = F3.*alpha .* (popdecs(r3,:) - popdecs(r4,:));
    
    mutants = popdecs + elite_term + diff_term1 + diff_term2;
    
    % 5. Adaptive crossover
    CR = 0.7 - 0.3 * alpha;
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 6. Boundary handling with reflection
    lb_viol = offspring < lb;
    ub_viol = offspring > ub;
    offspring(lb_viol) = 2*lb(lb_viol) - offspring(lb_viol);
    offspring(ub_viol) = 2*ub(ub_viol) - offspring(ub_viol);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub), lb);
end