% MATLAB Code
function [offspring] = updateFunc1036(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = max(f_max - f_min, eps);
    norm_fits = (popfits - f_min) ./ f_range;
    
    abs_cons = abs(cons);
    c_max = max(abs_cons);
    c_range = max(c_max, eps);
    norm_cons = abs_cons ./ c_range;
    
    % Weight calculation
    sigma_f = std(popfits) + eps;
    sigma_c = std(abs_cons) + eps;
    mean_f = mean(popfits);
    mean_c = mean(abs_cons);
    w = 1./(1 + exp((popfits - mean_f)/sigma_f)) .* 1./(1 + exp((abs_cons - mean_c)/sigma_c));
    
    % Elite pool (weighted selection)
    [~, sorted_idx] = sort(popfits);
    elite_num = max(2, floor(0.3*NP));
    elite_pool = popdecs(sorted_idx(1:elite_num), :);
    elite_weights = w(sorted_idx(1:elite_num));
    
    % Directional guidance
    dir = zeros(NP, D);
    for i = 1:NP
        diff = elite_pool - popdecs(i,:);
        weighted_diff = elite_weights' * diff;
        dir(i,:) = weighted_diff / (sum(elite_weights) + eps);
    end
    
    % Random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) randi(NP-1), idx); r1 = r1 + (r1 >= idx);
    r2 = arrayfun(@(i) randi(NP-1), idx); r2 = r2 + (r2 >= idx);
    r3 = arrayfun(@(i) randi(NP-1), idx); r3 = r3 + (r3 >= idx);
    
    % Adaptive parameters
    F = 0.5 + 0.3 * cos(pi * norm_fits);
    alpha = 0.5 * (1 - norm_cons);
    CR = 0.9 - 0.4 * norm_cons;
    
    % Mutation
    diff_vec = popdecs(r1,:) + F.*(popdecs(r2,:) - popdecs(r3,:));
    mutants = diff_vec + alpha .* dir;
    
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