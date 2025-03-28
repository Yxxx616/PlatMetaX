% MATLAB Code
function [offspring] = updateFunc1220(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps_val = 1e-10;
    
    % 1. Fitness-constraint balanced centroid
    weights = 1./(abs(popfits) + abs(cons) + eps_val;
    x_w = sum(popdecs .* weights, 1) ./ sum(weights);
    
    % 2. Adaptive directional vectors
    [~, sorted_idx] = sort(popfits);
    good_idx = sorted_idx(1:ceil(NP/2));
    bad_idx = sorted_idx(ceil(NP/2)+1:end);
    
    % Create direction vectors
    directions = zeros(NP, D);
    for i = 1:NP
        if cons(i) <= 0
            % Pull toward centroid if feasible
            directions(i,:) = x_w - popdecs(i,:);
        else
            % Differential vectors if infeasible
            r = randperm(NP,4);
            directions(i,:) = popdecs(r(1),:) - popdecs(r(2),:) + ...
                             popdecs(r(3),:) - popdecs(r(4),:);
        end
    end
    
    % 3. Dynamic scaling factor
    f_min = min(popfits);
    f_max = max(popfits);
    cv_abs = abs(cons);
    max_cv = max(cv_abs);
    
    F = 0.5 + 0.5 * ((popfits - f_min)./(f_max - f_min + eps_val)) .* ...
        (1 - cv_abs./(max_cv + eps_val));
    F = F(:, ones(1, D));
    
    % 4. Mutation with adaptive noise
    diversity = std(popdecs, 0, 1);
    sigma = 0.1 * diversity .* (1 - exp(-max_cv./(mean(cv_abs) + eps_val)));
    noise = randn(NP, D) .* sigma(ones(NP,1), :);
    
    mutants = x_w(ones(NP,1), :) + F .* directions + noise;
    
    % 5. Constraint-aware crossover
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    CR = 0.9 - 0.5 * (ranks/NP) .* (cv_abs/(max_cv + eps_val));
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