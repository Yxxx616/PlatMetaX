% MATLAB Code
function [offspring] = updateFunc1049(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    norm_f = (popfits - f_min) / (f_max - f_min + eps);
    norm_c = abs(cons) / (max(abs(cons)) + eps);
    
    % Select top 20% as elites with weighted contribution
    [sorted_f, sorted_idx] = sort(popfits);
    elite_num = max(2, floor(0.2*NP));
    elite_pool = popdecs(sorted_idx(1:elite_num), :);
    elite_fits = sorted_f(1:elite_num);
    
    % Calculate elite weights (fitness-based)
    weights = (f_max - elite_fits) / sum(f_max - elite_fits + eps);
    
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
    d_cons = sigma .* (popdecs(r1,:) - popdecs(r2,:));
    
    % Adaptive scaling factors
    F = 0.4 + 0.3 * cos(pi * norm_f);
    
    % Combined mutation
    mutants = popdecs + F.*d_elite + (1-F).*d_cons;
    
    % Dynamic crossover based on rank
    [~, ranks] = sort(popfits);
    norm_rank = (ranks-1)/(NP-1);
    CR = 0.9 * (1 - sqrt(norm_rank));
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling - reflection with adaptive perturbation
    lb_viol = offspring < lb;
    ub_viol = offspring > ub;
    rnd_factor = 0.2 + 0.6 * norm_f; % Better solutions get smaller perturbations
    offspring(lb_viol) = lb(lb_viol) + rnd_factor(lb_viol(:,1)) .* (ub(lb_viol)-lb(lb_viol));
    offspring(ub_viol) = ub(ub_viol) - rnd_factor(ub_viol(:,1)) .* (ub(ub_viol)-lb(ub_viol));
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub), lb);
end