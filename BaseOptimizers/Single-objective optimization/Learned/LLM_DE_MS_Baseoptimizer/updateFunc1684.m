% MATLAB Code
function [offspring] = updateFunc1684(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints (0 to 1, 0 is feasible)
    pos_cons = max(0, cons);
    norm_cons = pos_cons ./ (max(pos_cons) + eps);
    
    % Normalize fitness (0 to 1, 0 is best)
    norm_fits = (popfits - min(popfits)) ./ (max(popfits) - min(popfits) + eps);
    
    % Find best solution considering both fitness and constraints
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        temp = find(feasible_mask);
        x_best = popdecs(temp(best_idx),:);
    else
        [~, best_idx] = min(popfits + norm_cons*1e6);
        x_best = popdecs(best_idx,:);
    end
    
    % Initialize offspring
    offspring = zeros(NP, D);
    rand_vals = rand(NP, 5);
    
    for i = 1:NP
        % Select 4 distinct random indices different from i
        available = setdiff(1:NP, i);
        selected = available(randperm(length(available), 4));
        r1 = selected(1); r2 = selected(2); 
        r3 = selected(3); r4 = selected(4);
        
        % Adaptive parameters
        F_i = 0.5 * (1 - norm_cons(i)) + 0.3 * norm_fits(i);
        sigma_i = 0.2 * norm_cons(i);
        p_exploit = 0.7 * (1 - norm_fits(i));
        
        % Mutation strategy selection
        if rand_vals(i,1) < p_exploit
            % Exploitation: DE/current-to-best/1 with adaptive noise
            mutation = x_best + F_i * (popdecs(r1,:) - popdecs(r2,:)) + ...
                      sigma_i * (popdecs(r3,:) - popdecs(r4,:));
        else
            % Exploration: Opposition-based learning with differential
            mutation = lb + ub - popdecs(i,:) + F_i * (popdecs(r1,:) - popdecs(r2,:));
        end
        
        % Binomial crossover
        CR_i = 0.9 * (1 - norm_cons(i)) + 0.1 * norm_fits(i);
        j_rand = randi(D);
        mask = rand_vals(i,2:D+1) < CR_i;
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutation(mask);
    end
    
    % Smart boundary handling
    viol_low = offspring < lb;
    viol_high = offspring > ub;
    
    % For lower bound violations
    offspring(viol_low) = lb(viol_low) + 0.5 * rand(sum(viol_low(:)),1) .* ...
                         (popdecs(viol_low) - lb(viol_low));
    
    % For upper bound violations
    offspring(viol_high) = ub(viol_high) - 0.5 * rand(sum(viol_high(:)),1) .* ...
                          (ub(viol_high) - popdecs(viol_high));
    
    % Final bounds check
    offspring = max(min(offspring, ub), lb);
end