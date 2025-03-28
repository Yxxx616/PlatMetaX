% MATLAB Code
function [offspring] = updateFunc1039(popdecs, popfits, cons)
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
    
    % Calculate adaptive parameters
    sigma_f = std(popfits) + eps;
    sigma_c = std(abs_cons) + eps;
    mean_f = mean(popfits);
    mean_c = mean(abs_cons);
    
    % Fitness-adaptive scaling factor
    F = 0.5 * (1 + tanh((popfits - mean_f)./sigma_f));
    
    % Constraint-adaptive direction weight
    alpha = 0.5 * (1 - tanh((abs_cons - mean_c)./sigma_c));
    
    % Elite pool selection (top 20%)
    [~, sorted_idx] = sort(popfits);
    elite_num = max(2, floor(0.2*NP));
    elite_pool = popdecs(sorted_idx(1:elite_num), :);
    
    % Calculate weighted centroid
    weights = 1./(1:elite_num)';  % Higher weight for better solutions
    centroid = sum(weights .* elite_pool, 1) / sum(weights);
    
    % Direction vectors to centroid
    dir_vectors = centroid - popdecs;
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) randi(NP-1), idx); r1 = r1 + (r1 >= idx);
    r2 = arrayfun(@(i) randi(NP-1), idx); r2 = r2 + (r2 >= idx);
    r3 = arrayfun(@(i) randi(NP-1), idx); r3 = r3 + (r3 >= idx);
    
    % Mutation with adaptive parameters
    diff_vec = popdecs(r1,:) - popdecs(r2,:) + popdecs(r3,:);
    mutants = popdecs(r1,:) + F.*(popdecs(r2,:) - popdecs(r3,:)) + alpha.*dir_vectors;
    
    % Opposition-based learning
    opposite_pop = lb + ub - popdecs;
    
    % Adaptive crossover rate
    CR = 0.9 - 0.5 * norm_fits;
    
    % Crossover with jitter
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring with opposition-based diversity
    offspring = popdecs;
    use_opposition = rand(NP,1) < 0.3;
    offspring(use_opposition,:) = opposite_pop(use_opposition,:);
    offspring(~use_opposition & mask) = mutants(~use_opposition & mask);
    
    % Boundary handling with reflection
    lb_viol = offspring < lb;
    ub_viol = offspring > ub;
    offspring(lb_viol) = 2*lb(lb_viol) - offspring(lb_viol);
    offspring(ub_viol) = 2*ub(ub_viol) - offspring(ub_viol);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub), lb);
end