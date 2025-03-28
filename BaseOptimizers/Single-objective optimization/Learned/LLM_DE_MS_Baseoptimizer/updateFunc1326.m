% MATLAB Code
function [offspring] = updateFunc1326(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps_val = 1e-10;
    
    % 1. Constraint-weighted fitness calculation
    penalty = max(0, cons).^2;
    f_min = min(popfits);
    f_max = max(popfits);
    norm_f = (popfits - f_min) ./ (f_max - f_min + eps_val);
    weights = (1./(1 + penalty + eps_val)) .* (1 + norm_f);
    
    % 2. Elite selection (top 30%)
    [~, sorted_idx] = sort(weights, 'descend');
    elite_size = max(3, ceil(0.3 * NP));
    elite = popdecs(sorted_idx(1:elite_size), :);
    
    % 3. Improved mutation with dynamic scaling
    F_base = 0.5 + 0.3 * rand(NP, 1);
    F1 = F_base .* weights;
    F2 = 0.7 * F1;
    F3 = 0.5 * F1;
    
    % Generate indices for mutation
    e_idx = randi(elite_size, NP, 2);
    r_idx = randperm(NP);
    r1 = r_idx(1:NP);
    r2 = r_idx(mod(1:NP, NP) + 1);
    
    % Mutation operation (vectorized)
    mutants = popdecs + ...
              F1(:, ones(1,D)) .* (elite(e_idx(:,1),:) - popdecs) + ...
              F2(:, ones(1,D)) .* (elite(e_idx(:,2),:) - popdecs(r1,:)) + ...
              F3(:, ones(1,D)) .* (popdecs(r1,:) - popdecs(r2,:));
    
    % 4. Dynamic crossover rate
    [~, rank_idx] = sort(weights);
    rank = zeros(NP,1);
    rank(rank_idx) = (1:NP)';
    CR = 0.85 - 0.45*(rank/NP);
    CR = CR(:, ones(1,D));
    
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 5. Enhanced repair mechanism
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        x_best = popdecs(find(feasible_mask, best_idx, 'first'), :);
    else
        [~, best_idx] = min(cons);
        x_best = popdecs(best_idx, :);
    end
    
    % Adaptive repair strategy
    violate_mask = cons > 0;
    repair_prob = 0.8 - 0.3*(rank/NP);
    alpha = 0.3 + 0.5*rand(NP, 1);
    
    for i = 1:NP
        if violate_mask(i)
            if rand() < repair_prob(i)
                % Reflection towards best with adaptive step
                offspring(i,:) = x_best + alpha(i) * (x_best - offspring(i,:));
            else
                % Random reinitialization in feasible region if possible
                if any(feasible_mask)
                    offspring(i,:) = x_best + 0.5*(rand(1,D)-0.5).*(ub-lb);
                else
                    offspring(i,:) = lb + (ub - lb).*rand(1,D);
                end
            end
        end
    end
    
    % Bounds handling with reflection
    below_lb = offspring < lb;
    above_ub = offspring > ub;
    offspring(below_lb) = 2*lb(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub(above_ub) - offspring(above_ub);
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb), ub);
end