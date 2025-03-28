% MATLAB Code
function [offspring] = updateFunc1035(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = f_max - f_min + eps;
    norm_fits = (popfits - f_min) ./ f_range;
    
    sigma_cons = std(abs(cons)) + eps;
    w = exp(-abs(cons)/sigma_cons);
    
    % Select elite pool (top 30%)
    [~, sorted_idx] = sort(popfits);
    elite_num = max(1, floor(0.3*NP));
    elite_pool = popdecs(sorted_idx(1:elite_num), :);
    
    % Random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) randi(NP-1), idx); r1 = r1 + (r1 >= idx);
    r2 = arrayfun(@(i) randi(NP-1), idx); r2 = r2 + (r2 >= idx);
    r3 = arrayfun(@(i) randi(NP-1), idx); r3 = r3 + (r3 >= idx);
    elite_idx = randi(elite_num, NP, 1);
    
    % Adaptive parameters
    F = 0.5 * (1 + norm_fits);
    CR = 0.9 - 0.5 * norm_fits;
    
    % Mutation
    x_elite = elite_pool(elite_idx, :);
    elite_dir = x_elite - popdecs;
    diff_vec = popdecs(r1,:) + F.*(popdecs(r2,:) - popdecs(r3,:)) - popdecs;
    
    % Combined mutation with weights
    mutants = popdecs + w.*(1-norm_fits).*elite_dir + (1-w).*diff_vec;
    
    % Crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with reflection
    lb_viol = offspring < lb;
    ub_viol = offspring > ub;
    offspring(lb_viol) = 2*lb(lb_viol) - offspring(lb_viol);
    offspring(ub_viol) = 2*ub(ub_viol) - offspring(ub_viol);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub), lb);
end