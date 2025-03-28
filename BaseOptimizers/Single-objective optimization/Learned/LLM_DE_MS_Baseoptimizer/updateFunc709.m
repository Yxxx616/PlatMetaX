% MATLAB Code
function [offspring] = updateFunc709(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % 1. Normalize constraints and fitness
    c_norm = abs(cons) / (max(abs(cons)) + eps);
    f_norm = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    
    % 2. Select elite (best feasible) and best (overall)
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, elite_idx] = min(popfits + 1e6*max(0, cons));
        elite = popdecs(elite_idx, :);
    end
    [~, best_idx] = min(popfits);
    best = popdecs(best_idx, :);
    
    % 3. Generate random indices
    idx = randperm(NP);
    r1 = idx(1:NP);
    r2 = idx(mod(1:NP, NP)+1);
    r3 = idx(mod(2:NP+1, NP)+1);
    
    % 4. Constraint-driven perturbation
    Fc = 0.5 + 0.3 * c_norm;
    cons_diff = cons(r2) - cons(r3);
    phi = tanh(cons_diff); % smooth sign function
    v_c = popdecs(r1,:) + Fc(:, ones(1,D)) .* (popdecs(r2,:) - popdecs(r3,:)) .* phi(:, ones(1,D));
    
    % 5. Fitness-directed elite search
    Ff = 0.3 + 0.4 * f_norm;
    fit_diff = popfits(best_idx) - popfits(elite_idx);
    psi = tanh(fit_diff);
    v_f = elite + Ff(:, ones(1,D)) .* (best - elite) * psi;
    
    % 6. Adaptive combination
    alpha = abs(cons) ./ (abs(cons) + abs(popfits - popfits(best_idx)) + eps);
    mutant = alpha(:, ones(1,D)) .* v_c + (1-alpha(:, ones(1,D))) .* v_f;
    
    % 7. Adaptive crossover
    CR = 0.5 + 0.4 * (1:NP)'/NP;
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 8. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Final clipping to ensure feasibility
    offspring = max(min(offspring, ub_rep), lb_rep);
end