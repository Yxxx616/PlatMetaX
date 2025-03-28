% MATLAB Code
function [offspring] = updateFunc1034(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness
    f_min = min(popfits);
    f_max = max(popfits);
    if f_max - f_min < 1e-12
        norm_fits = zeros(NP, 1);
    else
        norm_fits = (popfits - f_min) / (f_max - f_min);
    end
    
    % Select top 20% as elite pool
    [~, sorted_idx] = sort(popfits);
    elite_num = max(1, floor(0.2*NP));
    elite_pool = popdecs(sorted_idx(1:elite_num), :);
    
    % Adaptive parameters
    F = 0.4 + 0.4 * norm_fits;
    CR = 0.9 - 0.5 * norm_fits;
    
    % Constraint-aware weights
    sigma_cons = std(abs(cons)) + 1e-12;
    w = exp(-abs(cons)/sigma_cons);
    
    % Random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) randi(NP-1), idx); r1 = r1 + (r1 >= idx);
    r2 = arrayfun(@(i) randi(NP-1), idx); r2 = r2 + (r2 >= idx);
    elite_idx = randi(elite_num, NP, 1);
    
    % Mutation
    x_elite = elite_pool(elite_idx, :);
    diff_vectors = popdecs(r1,:) - popdecs(r2,:);
    mutants = popdecs + F.*(x_elite - popdecs) + w.*diff_vectors;
    
    % Crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with midpoint reflection
    lb_viol = offspring < lb;
    ub_viol = offspring > ub;
    offspring(lb_viol) = (popdecs(lb_viol) + lb(lb_viol)) / 2;
    offspring(ub_viol) = (popdecs(ub_viol) + ub(ub_viol)) / 2;
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub), lb);
end