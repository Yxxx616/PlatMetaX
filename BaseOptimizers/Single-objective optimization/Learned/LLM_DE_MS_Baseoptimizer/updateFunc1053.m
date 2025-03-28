% MATLAB Code
function [offspring] = updateFunc1053(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Compute elite direction with fitness weights
    sigma_f = std(popfits) + eps;
    weights = exp(-popfits/sigma_f);
    weights = weights / sum(weights);
    elite_vec = weights' * popdecs;
    
    % 2. Constraint-aware scaling factors
    max_cons = max(abs(cons)) + eps;
    alpha = 1 - abs(cons)/max_cons;
    
    % 3. Opposition-based vectors
    opp_vec = lb + ub - popdecs;
    
    % 4. Differential direction vectors
    idx = randperm(NP);
    r1 = idx(1:NP);
    r2 = idx(mod(1:NP, NP)+1);
    r3 = idx(mod(2:NP+1, NP)+1);
    r4 = idx(mod(3:NP+2, NP)+1);
    diff_vec = popdecs(r1,:) - popdecs(r2,:) + popdecs(r3,:) - popdecs(r4,:);
    
    % 5. Adaptive scaling factors
    F1 = 0.5 + 0.3*rand(NP,1);
    F2 = 0.3 + 0.2*rand(NP,1);
    F3 = 0.2 + 0.1*rand(NP,1);
    
    % 6. Combined mutation
    mutants = popdecs + F1.*(elite_vec - popdecs) + ...
              F2.*alpha.*(opp_vec - popdecs) + ...
              F3.*diff_vec;
    
    % 7. Adaptive crossover
    CR = 0.5 + 0.3*alpha;
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Boundary handling with reflection
    lb_viol = offspring < lb;
    ub_viol = offspring > ub;
    offspring(lb_viol) = 2*lb(lb_viol) - offspring(lb_viol);
    offspring(ub_viol) = 2*ub(ub_viol) - offspring(ub_viol);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub), lb);
end