% MATLAB Code
function [offspring] = updateFunc877(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Find best feasible solution
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        x_best = popdecs(feasible_mask(best_idx), :);
    else
        [~, best_idx] = min(cons);
        x_best = popdecs(best_idx, :);
    end
    
    % Compute selection probabilities based on constraints
    alpha = 5; % Constraint weight parameter
    exp_cons = exp(-alpha * cons);
    p = exp_cons / sum(exp_cons);
    
    % Compute statistics for adaptive scaling factor
    mu_f = mean(popfits);
    sigma_f = std(popfits) + 1e-12;
    F_base = 0.5;
    
    offspring = zeros(NP, D);
    for i = 1:NP
        % Adaptive scaling factor based on fitness
        z_score = (popfits(i) - mu_f) / sigma_f;
        F_i = F_base + 0.3 / (1 + exp(-z_score)); % Sigmoid mapping
        
        % Select 4 distinct individuals using weighted probabilities
        candidates = setdiff(1:NP, i);
        idx = randsample(candidates, 4, true, p(candidates));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4);
        
        % Hybrid mutation
        mutant = popdecs(r1,:) + F_i * (...
            (x_best - popdecs(r2,:)) + ...
            (popdecs(r3,:) - popdecs(r4,:)));
        
        % Adaptive crossover based on constraint violation
        CR = 0.9 - 0.4 * (cons(i) - min(cons)) / (max(cons) - min(cons) + 1e-12);
        mask = rand(1,D) < CR;
        j_rand = randi(D);
        mask(j_rand) = true;
        
        % Create offspring
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutant(mask);
    end
    
    % Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % For feasible solutions: reflection
    feasible = repmat(cons <= 0, 1, D);
    offspring(feasible & (offspring < lb_rep)) = 2*lb_rep(feasible & (offspring < lb_rep)) - offspring(feasible & (offspring < lb_rep));
    offspring(feasible & (offspring > ub_rep)) = 2*ub_rep(feasible & (offspring > ub_rep)) - offspring(feasible & (offspring > ub_rep));
    
    % For infeasible solutions: random reset
    infeasible = ~feasible;
    rand_mask = rand(NP,D) < 0.1; % 10% chance to reset
    reset_dims = infeasible & rand_mask;
    rand_vals = repmat(lb, NP, 1) + rand(NP,D).*repmat(ub-lb, NP, 1);
    offspring(reset_dims) = rand_vals(reset_dims);
    
    % Final clamping
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Elite preservation: keep best solution unchanged
    if any(feasible_mask)
        offspring(best_idx,:) = x_best;
    end
end