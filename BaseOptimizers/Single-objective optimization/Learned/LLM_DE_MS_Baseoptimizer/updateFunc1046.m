% MATLAB Code
function [offspring] = updateFunc1046(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Find best individual considering both fitness and constraints
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        x_best = popdecs(feasible_mask(best_idx), :);
    else
        [~, best_idx] = min(popfits + 1e6*max(0,cons));
        x_best = popdecs(best_idx, :);
    end
    
    % Select top 20% as elites
    [~, sorted_idx] = sort(popfits);
    elite_num = max(2, floor(0.2*NP));
    elite_pool = popdecs(sorted_idx(1:elite_num), :);
    
    % Calculate weights based on fitness rank
    ranks = 1:elite_num;
    weights = (elite_num - ranks + 1) / sum(1:elite_num);
    
    % Normalize fitness and constraints
    norm_f = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    norm_c = abs(cons) / (max(abs(cons)) + eps);
    
    % Base parameters
    F = 0.5;
    alpha = 0.3;
    beta = 0.2;
    
    % Elite direction component (vectorized)
    elite_diff = reshape(elite_pool, elite_num, 1, D) - reshape(popdecs, 1, NP, D);
    weighted_diff = sum(reshape(weights, elite_num, 1, 1) .* elite_diff, 1);
    d_elite = squeeze(weighted_diff)';
    
    % Constraint-aware perturbation
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2(r1 == r2) = randi(NP, sum(r1 == r2), 1);
    end
    rand_vec = rand(NP, D);
    sigma = 1 + tanh(norm_c);
    d_cons = sigma .* rand_vec .* (popdecs(r1,:) - popdecs(r2,:));
    
    % Best individual attraction
    eta = 0.1 * (1 - norm_f);
    d_best = eta .* (x_best - popdecs);
    
    % Mutation
    mutants = popdecs + F*d_elite + alpha*d_cons + beta*d_best;
    
    % Dynamic crossover
    CR_base = 0.9;
    CR = CR_base * (1 - norm_f);
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling - reflection
    lb_viol = offspring < lb;
    ub_viol = offspring > ub;
    offspring(lb_viol) = 2*lb(lb_viol) - offspring(lb_viol);
    offspring(ub_viol) = 2*ub(ub_viol) - offspring(ub_viol);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub), lb);
end