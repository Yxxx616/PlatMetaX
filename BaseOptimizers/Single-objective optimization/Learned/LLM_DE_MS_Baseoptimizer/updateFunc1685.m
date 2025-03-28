% MATLAB Code
function [offspring] = updateFunc1685(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints (positive values indicate violation)
    pos_cons = max(0, cons);
    max_cons = max(pos_cons) + eps;
    norm_cons = pos_cons / max_cons;
    
    % Normalize fitness (0 is best)
    min_fit = min(popfits);
    max_fit = max(popfits);
    norm_fits = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    % Find best solution (considering both fitness and constraints)
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        temp = find(feasible);
        x_best = popdecs(temp(best_idx), :);
    else
        [~, best_idx] = min(popfits + pos_cons*1e6);
        x_best = popdecs(best_idx, :);
    end
    
    % Initialize offspring
    offspring = zeros(NP, D);
    
    % Pre-generate random numbers for vectorization
    rand_vals = rand(NP, D+2);
    rand_idx = randi(NP, NP, 4);
    
    for i = 1:NP
        % Ensure distinct indices (different from i)
        idx_pool = setdiff(1:NP, i);
        r = idx_pool(randperm(length(idx_pool), 4));
        r1 = r(1); r2 = r(2); r3 = r(3); r4 = r(4);
        
        % Adaptive parameters
        F = 0.5 * (1 - norm_cons(i)) + 0.1;
        sigma = 0.3 * (1 - norm_fits(i));
        CR = 0.9 * (1 - norm_cons(i)) + 0.1;
        
        % Constraint-aware mutation
        mutation = popdecs(r1,:) + F * (popdecs(r2,:) - popdecs(r3,:)) + ...
                  sigma * (x_best - popdecs(i,:));
        
        % Binomial crossover
        j_rand = randi(D);
        mask = rand_vals(i, 1:D) < CR;
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutation(mask);
        
        % Smart boundary handling
        viol_low = offspring(i,:) < lb;
        viol_high = offspring(i,:) > ub;
        
        if any(viol_low)
            offspring(i,viol_low) = (lb(viol_low) + popdecs(i,viol_low))/2 + ...
                                   rand(1,sum(viol_low)).*(popdecs(r1,viol_low) - lb(viol_low))/2;
        end
        
        if any(viol_high)
            offspring(i,viol_high) = (ub(viol_high) + popdecs(i,viol_high))/2 - ...
                                    rand(1,sum(viol_high)).*(ub(viol_high) - popdecs(r2,viol_high))/2;
        end
    end
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub), lb);
end