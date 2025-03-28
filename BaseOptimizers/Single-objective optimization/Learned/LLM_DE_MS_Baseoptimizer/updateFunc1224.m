% MATLAB Code
function [offspring] = updateFunc1224(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps_val = 1e-10;
    
    % 1. Elite selection and centroid calculation
    [~, sorted_idx] = sort(popfits);
    elite_num = max(1, floor(0.3 * NP));
    elites = popdecs(sorted_idx(1:elite_num), :);
    centroid = mean(elites, 1);
    
    % 2. Adaptive scaling factors
    f_min = min(popfits);
    f_max = max(popfits);
    cv_abs = abs(cons);
    max_cv = max(cv_abs);
    
    F = 0.5 * (1 + cv_abs./(max_cv + eps_val)) .* ...
               (1 - (popfits - f_min)./(f_max - f_min + eps_val));
    F = F(:, ones(1, D));
    
    % 3. Direction vectors
    directions = zeros(NP, D);
    for i = 1:NP
        if cons(i) <= 0
            directions(i,:) = centroid - popdecs(i,:);
        else
            candidates = setdiff(1:NP, i);
            r = candidates(randperm(length(candidates), 4));
            directions(i,:) = popdecs(r(1),:) - popdecs(r(2),:) + ...
                             popdecs(r(3),:) - popdecs(r(4),:);
        end
    end
    
    % 4. Mutation with adaptive noise
    diversity = std(popdecs, 0, 1);
    sigma = 0.1 * diversity;
    noise = randn(NP, D) .* sigma(ones(NP,1), :);
    
    mutants = popdecs + F .* directions + noise;
    
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