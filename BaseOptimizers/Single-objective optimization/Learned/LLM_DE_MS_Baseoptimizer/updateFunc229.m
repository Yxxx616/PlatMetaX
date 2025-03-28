% MATLAB Code
function [offspring] = updateFunc229(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Boundary definitions
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Enhanced constraint processing
    abs_cons = abs(cons);
    max_con = max(abs_cons) + eps;
    min_con = min(abs_cons);
    weights = 1 ./ (1 + exp(5 * (abs_cons - min_con) ./ (max_con - min_con + eps)));
    
    % Improved fitness normalization
    median_fit = median(popfits);
    mad_fit = median(abs(popfits - median_fit)) + eps;
    norm_fits = (popfits - median_fit) ./ mad_fit;
    
    % Population diversity using median absolute deviation
    diversity = median(std(popdecs)) + eps;
    
    % Elite selection with combined fitness and constraints
    penalty = 1e6;
    combined_scores = popfits + penalty * abs_cons;
    [~, elite_idx] = min(combined_scores);
    elite = popdecs(elite_idx, :);
    
    % Best feasible solution selection with fallback
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        temp = popdecs(feasible_mask, :);
        best_feas = temp(best_idx, :);
    else
        [~, min_con_idx] = min(abs_cons);
        best_feas = popdecs(min_con_idx, :);
    end
    
    % Adaptive scaling factors
    F1 = 0.5 * (1 - tanh(norm_fits));  % Fitness-based
    F2 = 0.3 * (1 - weights);          % Constraint-based
    F3 = 0.2 * diversity/D;            % Diversity-based
    
    % Vectorized mutation with improved selection
    for i = 1:NP
        % Select distinct random individuals excluding current and elite
        candidates = setdiff(1:NP, [i, elite_idx]);
        idxs = candidates(randperm(length(candidates), 2));
        
        % Elite-directed differential component
        diff1 = popdecs(idxs(1),:) - popdecs(idxs(2),:);
        
        % Constraint-aware perturbation
        diff2 = best_feas - popdecs(i,:);
        
        % Combine components with adaptive weights
        offspring(i,:) = elite + F1(i)*diff1 + F2(i)*diff2 + F3*randn(1,D);
    end
    
    % Smart boundary handling with adaptive reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    
    r = rand(NP,D);
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + r.*(ub - lb)).*out_low + ...
               (ub - r.*(ub - lb)).*out_high;
    
    % Final boundary enforcement
    offspring = max(min(offspring, ub), lb);
end