% MATLAB Code
function [offspring] = updateFunc878(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Find best solution (feasible first, then least infeasible)
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        x_best = popdecs(feasible_mask(best_idx), :);
    else
        [~, best_idx] = min(cons);
        x_best = popdecs(best_idx, :);
    end
    
    % Normalize fitness and constraints for adaptation
    mu_f = mean(popfits);
    sigma_f = std(popfits) + 1e-12;
    norm_f = (popfits - mu_f) / sigma_f;
    
    c_min = min(cons);
    c_max = max(cons);
    norm_c = (cons - c_min) / (c_max - c_min + 1e-12);
    
    % Initialize offspring
    offspring = zeros(NP, D);
    
    for i = 1:NP
        % Adaptive parameters
        F_i = 0.5 * (1 + tanh(norm_f(i))); % [0,1] range
        CR_i = 0.9 - 0.5 * norm_c(i);     % [0.4,0.9] range
        
        % Select two distinct random vectors
        candidates = setdiff(1:NP, [i, best_idx]);
        idx = candidates(randperm(length(candidates), 2));
        r1 = idx(1); r2 = idx(2);
        
        % Mutation
        direction = (x_best - popdecs(i,:)) + (popdecs(r1,:) - popdecs(r2,:));
        mutant = popdecs(i,:) + F_i * direction;
        
        % Crossover
        mask = rand(1,D) < CR_i;
        j_rand = randi(D);
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutant(mask);
    end
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection for feasible, random reset for infeasible
    feasible = repmat(feasible_mask, 1, D);
    offspring(feasible & (offspring < lb_rep)) = 2*lb_rep(feasible & (offspring < lb_rep)) - offspring(feasible & (offspring < lb_rep));
    offspring(feasible & (offspring > ub_rep)) = 2*ub_rep(feasible & (offspring > ub_rep)) - offspring(feasible & (offspring > ub_rep));
    
    infeasible = ~feasible;
    rand_vals = lb_rep + rand(NP,D).*(ub_rep - lb_rep);
    offspring(infeasible & (offspring < lb_rep | offspring > ub_rep)) = rand_vals(infeasible & (offspring < lb_rep | offspring > ub_rep));
    
    % Ensure best solution is preserved
    if any(feasible_mask)
        offspring(best_idx,:) = x_best;
    end
end