% MATLAB Code
function [offspring] = updateFunc1660(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints and fitness
    abs_cons = abs(cons);
    norm_cons = abs_cons / (max(abs_cons) + eps);
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    
    % Find best solution considering both fitness and constraints
    combined_score = popfits + 100*norm_cons;
    [~, best_idx] = min(combined_score);
    x_best = popdecs(best_idx, :);
    
    % Initialize offspring
    offspring = zeros(NP, D);
    
    for i = 1:NP
        % Select 4 distinct random indices different from i
        candidates = setdiff(1:NP, i);
        r_idx = candidates(randperm(length(candidates), 4));
        r1 = r_idx(1); r2 = r_idx(2); r3 = r_idx(3); r4 = r_idx(4);
        
        % Adaptive parameters
        F1 = 0.7 * (1 - norm_cons(i));
        F2 = 0.5 * norm_fits(i);
        F3 = 0.3 * norm_cons(i);
        CR = 0.9 - 0.5 * norm_cons(i);
        
        % Weight factors
        w_fit = popfits(r2) / (popfits(r1) + popfits(r2) + eps);
        w_con = 1 + norm_cons(r3) / (norm_cons(r4) + eps);
        
        % Composite mutation strategy
        mutation = popdecs(i,:) + ...
                  F1 .* (x_best - popdecs(i,:)) + ...
                  F2 .* (popdecs(r1,:) - popdecs(r2,:)) .* w_fit + ...
                  F3 .* (popdecs(r3,:) - popdecs(r4,:)) .* w_con;
        
        % Binomial crossover
        j_rand = randi(D);
        mask = rand(1, D) < CR;
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutation(mask);
    end
    
    % Enhanced boundary handling using best solution guidance
    viol_low = offspring < lb;
    viol_high = offspring > ub;
    
    offspring(viol_low) = lb(viol_low) + rand(sum(viol_low(:)),1) .* ...
                         (x_best(viol_low) - lb(viol_low));
    offspring(viol_high) = ub(viol_high) - rand(sum(viol_high(:)),1) .* ...
                          (ub(viol_high) - x_best(viol_high));
    
    % Small random perturbation (2% probability)
    rand_mask = rand(NP,D) < 0.02;
    offspring(rand_mask) = lb(rand_mask) + rand(sum(rand_mask(:)),1).*...
                         (ub(rand_mask)-lb(rand_mask));
end