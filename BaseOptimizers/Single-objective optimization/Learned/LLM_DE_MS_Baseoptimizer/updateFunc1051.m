% MATLAB Code
function [offspring] = updateFunc1051(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Elite selection with exponential weighting
    [~, sorted_idx] = sort(popfits);
    elite_num = max(2, floor(0.2*NP));
    elite_pool = popdecs(sorted_idx(1:elite_num), :);
    
    % Exponential weights based on rank
    ranks = (1:elite_num)';
    weights = exp(-0.5 * ranks);
    weights = weights / sum(weights);
    
    % Weighted elite vector (vectorized)
    elite_vec = sum(weights .* elite_pool, 1);
    
    % 2. Constraint-aware scaling factors
    mu_c = mean(abs(cons));
    sigma_c = std(abs(cons)) + eps;
    beta = 0.5 * (1 + tanh((abs(cons) - mu_c)/sigma_c));
    
    % 3. Directional perturbation based on constraints
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2(r1 == r2) = randi(NP, sum(r1 == r2), 1);
    end
    dir_vec = sign(cons) .* (popdecs(r1,:) - popdecs(r2,:));
    
    % 4. Random perturbation
    xi = randn(NP, D);
    
    % 5. Combined mutation (vectorized)
    F = 0.8;
    gamma = 0.1;
    mutants = popdecs + F*(elite_vec - popdecs) + beta.*dir_vec + gamma*xi;
    
    % 6. Adaptive crossover
    CR = 0.7 + 0.2*(1 - beta);
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Boundary handling with reflection
    lb_viol = offspring < lb;
    ub_viol = offspring > ub;
    
    reflect_lb = 2*lb - offspring;
    reflect_ub = 2*ub - offspring;
    
    offspring(lb_viol) = reflect_lb(lb_viol);
    offspring(ub_viol) = reflect_ub(ub_viol);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub), lb);
end