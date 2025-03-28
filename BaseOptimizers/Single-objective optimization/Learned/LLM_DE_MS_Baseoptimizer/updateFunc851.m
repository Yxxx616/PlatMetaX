% MATLAB Code
function [offspring] = updateFunc851(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Elite selection
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Best solution
    [~, best_idx] = min(popfits);
    best = popdecs(best_idx, :);
    
    % Feasible mean
    if any(feasible)
        feasible_mean = mean(popdecs(feasible,:), 1);
    else
        feasible_mean = mean(popdecs, 1);
    end
    
    % Normalized weights
    f_norm = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    c_norm = (cons - min(cons)) / (max(cons) - min(cons) + eps);
    w_f = 1 - f_norm;
    w_c = 1 - c_norm;
    alpha = 0.6 * w_f + 0.4 * w_c;
    CR = 0.9 * w_f + 0.1 * w_c;
    
    % Random pairs for diversity
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2(r1 == r2) = randi(NP, sum(r1 == r2), 1);
    end
    
    % Mutation vectors
    elite_vec = repmat(elite, NP, 1) - popdecs;
    best_vec = repmat(best, NP, 1) - popdecs;
    feasible_diff = repmat(feasible_mean, NP, 1) - popdecs;
    random_diff = popdecs(r1,:) - popdecs(r2,:);
    
    % Combined mutation with adaptive weights
    beta = 0.7;
    mutant = popdecs + alpha .* (elite_vec + beta.*best_vec + (1-beta).*feasible_diff) + ...
             (1-alpha) .* random_diff;
    
    % Crossover with adaptive CR
    mask = rand(NP, D) < CR(:, ones(1, D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    offspring = max(min(offspring, ub_rep), lb_rep);
end