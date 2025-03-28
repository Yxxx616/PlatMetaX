% MATLAB Code
function [offspring] = updateFunc1222(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps_val = 1e-10;
    
    % 1. Weighted centroid calculation
    weights = 1./(abs(popfits) + abs(cons) + eps_val);
    x_w = sum(popdecs .* weights, 1) ./ sum(weights);
    
    % 2. Adaptive direction vectors
    directions = zeros(NP, D);
    for i = 1:NP
        if cons(i) <= 0
            directions(i,:) = x_w - popdecs(i,:);
        else
            candidates = setdiff(1:NP, i);
            r = candidates(randperm(length(candidates), 4));
            directions(i,:) = popdecs(r(1),:) - popdecs(r(2),:) + ...
                             popdecs(r(3),:) - popdecs(r(4),:);
        end
    end
    
    % 3. Dynamic scaling factor
    f_min = min(popfits);
    f_max = max(popfits);
    cv_abs = abs(cons);
    max_cv = max(cv_abs);
    
    F = 0.5 + 0.3 * (1 - cv_abs./(max_cv + eps_val)) .* ...
               (1 - (popfits - f_min)./(f_max - f_min + eps_val));
    F = F(:, ones(1, D));
    
    % 4. Constraint-guided mutation with adaptive noise
    diversity = std(popdecs, 0, 1);
    sigma = 0.1 * diversity;
    noise = randn(NP, D) .* sigma(ones(NP,1), :);
    
    mutants = x_w(ones(NP,1), :) + F .* directions + noise;
    
    % 5. Adaptive crossover
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    CR = 0.9 * (1 - ranks/NP) .* (1 - cv_abs/(max_cv + eps_val));
    CR = CR(:, ones(1, D));
    
    % Perform crossover
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    
    % Final bounds check
    offspring = min(max(offspring, lb), ub);
end