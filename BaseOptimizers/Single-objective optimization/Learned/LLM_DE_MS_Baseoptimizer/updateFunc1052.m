% MATLAB Code
function [offspring] = updateFunc1052(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Fitness-weighted elite vector
    sigma_f = std(popfits) + eps;
    weights = exp(-popfits/sigma_f);
    weights = weights / sum(weights);
    elite_vec = weights' * popdecs;
    
    % 2. Constraint-aware scaling factors
    max_cons = max(abs(cons)) + eps;
    alpha = 1 - abs(cons)/max_cons;
    
    % 3. Directional perturbation
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2(r1 == r2) = randi(NP, sum(r1 == r2), 1);
    end
    dir_vec = sign(cons) .* (popdecs(r1,:) - popdecs(r2,:));
    
    % 4. Random perturbation
    z = randn(NP, D);
    
    % 5. Combined mutation
    F = 0.7;
    gamma = 0.15;
    mutants = popdecs + F*(elite_vec - popdecs) + alpha.*dir_vec + gamma*z;
    
    % 6. Adaptive crossover
    CR = 0.5 + 0.3*alpha;
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