% MATLAB Code
function [offspring] = updateFunc1321(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps_val = 1e-10;
    
    % 1. Constraint-weighted fitness calculation
    feasible_mask = cons <= 0;
    penalty = max(0, cons).^2;
    f_min = min(popfits);
    f_max = max(popfits);
    norm_f = (popfits - f_min) ./ (f_max - f_min + eps_val);
    weights = (1./(1 + penalty + eps_val)) .* (1 + norm_f);
    
    % 2. Elite selection (top 30%)
    [~, sorted_idx] = sort(weights, 'descend');
    elite_size = ceil(0.3 * NP);
    elite = popdecs(sorted_idx(1:elite_size), :);
    
    % 3. Directional mutation with adaptive F
    F_base = 0.5 * weights ./ (max(weights) + eps_val);
    F = F_base(:, ones(1, D));
    
    % Generate indices for mutation
    e_idx = randi(elite_size, NP, 2);
    r_idx = randperm(NP);
    r1 = r_idx(1:NP);
    r2 = r_idx(mod(1:NP, NP) + 1);
    
    % Mutation operation
    mutants = popdecs + F .* (elite(e_idx(:,1),:) - popdecs) + ...
              F .* (elite(e_idx(:,2),:) - popdecs(r1,:)) + ...
              F .* (popdecs(r1,:) - popdecs(r2,:));
    
    % 4. Rank-based crossover
    [~, rank_idx] = sort(weights);
    rank = zeros(NP,1);
    rank(rank_idx) = (1:NP)';
    CR = 0.9 - 0.5*(rank/NP);
    CR = CR(:, ones(1,D));
    
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 5. Feasibility-guided repair
    new_cons = cons; % Placeholder for new constraints (would need actual constraint evaluation)
    violate_mask = new_cons > 0;
    
    % Find best feasible solution
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        x_best = popdecs(find(feasible_mask, best_idx), :);
    else
        [~, best_idx] = min(cons);
        x_best = popdecs(best_idx, :);
    end
    
    % Repair strategy
    repair_choice = rand(NP, 1) < 0.7;
    for i = 1:NP
        if violate_mask(i)
            if repair_choice(i)
                % Reflection towards best
                offspring(i,:) = x_best + 0.5*(x_best - offspring(i,:));
            else
                % Random reinitialization
                offspring(i,:) = lb + (ub - lb).*rand(1,D);
            end
        end
    end
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb), ub);
end