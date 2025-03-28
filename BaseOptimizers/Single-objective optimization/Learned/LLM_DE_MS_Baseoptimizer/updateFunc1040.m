% MATLAB Code
function [offspring] = updateFunc1040(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Sort population by fitness
    [~, sorted_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(sorted_idx) = 1:NP;
    
    % Elite selection (top 30%)
    elite_num = max(2, floor(0.3*NP));
    elite_pool = popdecs(sorted_idx(1:elite_num), :);
    
    % Calculate weighted centroid
    weights = 1./(1 + (1:elite_num)');
    centroid = sum(weights .* elite_pool, 1) / sum(weights);
    
    % Direction vectors to centroid
    dir_vectors = centroid - popdecs;
    
    % Constraint adaptation
    abs_cons = abs(cons);
    mean_c = mean(abs_cons);
    sigma_c = std(abs_cons) + eps;
    beta = 0.5 * (1 + tanh((abs_cons - mean_c)./sigma_c));
    
    % Adaptive scaling factors
    F = 0.5 + 0.3 * sin(pi * ranks / NP);
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) randi(NP-1), idx); r1 = r1 + (r1 >= idx);
    r2 = arrayfun(@(i) randi(NP-1), idx); r2 = r2 + (r2 >= idx);
    r3 = arrayfun(@(i) randi(NP-1), idx); r3 = r3 + (r3 >= idx);
    
    % Mutation with adaptive parameters
    mutants = popdecs(r1,:) + F.*(popdecs(r2,:) - popdecs(r3,:)) + beta.*dir_vectors;
    
    % Adaptive crossover rate
    CR = 0.9 - 0.4 * (ranks/NP);
    
    % Crossover with jitter
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