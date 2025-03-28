% MATLAB Code
function [offspring] = updateFunc1059(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Fitness-based weights for elite direction
    mean_f = mean(popfits);
    sigma_f = std(popfits) + eps;
    weights = 1./(1 + exp((popfits - mean_f)/sigma_f));
    elite_vec = (weights' * popdecs) / sum(weights);
    
    % 2. Rank-based indices
    [~, rank_idx] = sort(popfits);
    r1 = rank_idx(randperm(NP));
    r2 = rank_idx(randperm(NP));
    r3 = rank_idx(randperm(NP));
    r4 = rank_idx(randperm(NP));
    
    % 3. Constraint-aware adaptation
    max_cons = max(abs(cons)) + eps;
    alpha = 0.5 + 0.5*tanh(abs(cons)/max_cons);
    
    % 4. Adaptive scaling factors
    F1 = 0.6 + 0.2*rand(NP,1);
    F2 = 0.4 + 0.3*rand(NP,1);
    F3 = 0.1 + 0.2*rand(NP,1);
    
    % 5. Enhanced mutation
    elite_term = F1.*alpha .* (elite_vec - popdecs);
    diff_term1 = F2.*(1-alpha) .* (popdecs(r1,:) - popdecs(r2,:));
    diff_term2 = F3.*alpha .* (popdecs(r3,:) - popdecs(r4,:));
    mutants = popdecs + elite_term + diff_term1 + diff_term2;
    
    % 6. Dynamic crossover
    CR = 0.7 - 0.3*alpha;
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Boundary handling with reflection
    lb_viol = offspring < lb;
    ub_viol = offspring > ub;
    offspring(lb_viol) = 2*lb(lb_viol) - offspring(lb_viol);
    offspring(ub_viol) = 2*ub(ub_viol) - offspring(ub_viol);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub), lb);
end