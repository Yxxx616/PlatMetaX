% MATLAB Code
function [offspring] = updateFunc1047(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    norm_f = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    norm_c = abs(cons) / (max(abs(cons)) + eps;
    
    % Select top 20% as elites
    [~, sorted_idx] = sort(popfits);
    elite_num = max(2, floor(0.2*NP));
    elite_pool = popdecs(sorted_idx(1:elite_num), :);
    
    % Calculate weights (linear ranking)
    ranks = 1:elite_num;
    weights = (elite_num - ranks + 1) / sum(1:elite_num);
    
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
    sigma = 1 + tanh(norm_c);
    d_cons = sigma .* rand(NP, D) .* (popdecs(r1,:) - popdecs(r2,:));
    
    % Adaptive scaling factors
    F_base = 0.6;
    F = F_base * (1 - norm_f);
    
    % Mutation
    mutants = popdecs + F.*d_elite + 0.4*d_cons;
    
    % Dynamic crossover
    [~, ranks] = sort(popfits);
    norm_rank = (ranks-1)/(NP-1);
    CR = 0.9 * (1 - norm_rank);
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