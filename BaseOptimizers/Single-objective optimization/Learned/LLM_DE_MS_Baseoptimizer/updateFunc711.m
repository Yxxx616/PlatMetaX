% MATLAB Code
function [offspring] = updateFunc711(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Normalize constraints and fitness
    c_norm = abs(cons) / (max(abs(cons)) + eps);
    f_norm = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    
    % Identify elite (best feasible) and best (overall)
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, elite_idx] = min(c_norm);
        elite = popdecs(elite_idx, :);
    end
    [~, best_idx] = min(popfits);
    best = popdecs(best_idx, :);
    
    % Generate random indices
    idx = randperm(NP);
    r1 = idx(1:NP);
    r2 = idx(mod(1:NP, NP)+1);
    r3 = idx(mod(2:NP+1, NP)+1);
    r4 = idx(mod(3:NP+2, NP)+1);
    r5 = idx(mod(4:NP+3, NP)+1);
    r6 = idx(mod(5:NP+4, NP)+1);
    
    % Adaptive parameters
    alpha = 0.5 * (1 + tanh(f_norm));  % Fitness-based weight
    beta = 0.3 * (1 + tanh(c_norm));   % Constraint-based weight
    F = 0.5 * (1 + cos(pi * 711/1000)); % Current iteration is 711
    
    % Mutation strategies
    mutant1 = elite + F * (popdecs(r1,:) - popdecs(r2,:));
    mutant2 = popdecs + alpha.*(best - popdecs) + F*(popdecs(r3,:) - popdecs(r4,:));
    mutant3 = popdecs + beta.*c_norm.*(popdecs(r5,:) - popdecs(r6,:));
    
    % Select mutation strategy probabilistically
    rand_vals = rand(NP,1);
    mask1 = rand_vals < alpha;
    mask2 = (~mask1) & (rand_vals < (alpha + beta));
    mask3 = ~(mask1 | mask2);
    
    mutant = mask1.*mutant1 + mask2.*mutant2 + mask3.*mutant3;
    
    % Adaptive crossover
    CR = 0.7 + 0.2*(1:NP)'/NP;
    mask_cr = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask_cr = mask_cr | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask_cr) = mutant(mask_cr);
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Final clipping to ensure feasibility
    offspring = max(min(offspring, ub_rep), lb_rep);
end