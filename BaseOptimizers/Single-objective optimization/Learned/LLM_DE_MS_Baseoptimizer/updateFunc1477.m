% MATLAB Code
function [offspring] = updateFunc1477(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Process constraint violations
    cv = max(0, cons);
    w = cv ./ (max(cv) + eps);
    
    % Select elite individual (minimizing fitness + penalty)
    [~, elite_idx] = min(popfits + 1e6*cv);
    x_elite = popdecs(elite_idx, :);
    
    % Generate random indices (distinct from current and each other)
    r = zeros(NP, 3);
    for i = 1:NP
        candidates = setdiff(1:NP, i);
        r(i,:) = candidates(randperm(length(candidates), 3));
    end
    
    % Calculate direction vectors
    elite_dir = x_elite - popdecs;
    opposition = (lb + ub - popdecs(r(:,1),:) - popdecs;
    rand_diff = popdecs(r(:,2),:) - popdecs(r(:,3),:);
    
    % Adaptive scaling factors
    F_elite = 0.8*(1 - w) + 0.1*rand(NP,1);
    F_opp = 0.4*(1 - w.^2).*rand(NP,1);
    F_rand = 0.2*(w.^2).*rand(NP,1);
    
    % Composite mutation
    mutants = popdecs + F_elite.*elite_dir + F_opp.*opposition + F_rand.*rand_diff;
    
    % Boundary handling
    mutants = min(max(mutants, lb), ub);
    
    % Constraint-aware crossover
    CR = 0.9 * (1 - sqrt(w));
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Local refinement for top solutions
    [~, sorted_idx] = sort(popfits + 1e6*cv);
    refine_N = max(1, round(0.1*NP));
    refine_idx = sorted_idx(1:refine_N);
    
    sigma = 0.05 * (1-w(refine_idx)) .* (ub-lb) .* (1 + 0.1*randn(refine_N,D));
    local_search = popdecs(refine_idx,:) + sigma;
    local_search = min(max(local_search, lb), ub);
    
    % Replace with refined solutions
    offspring(refine_idx,:) = local_search;
end