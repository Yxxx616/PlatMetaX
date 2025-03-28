% MATLAB Code
function [offspring] = updateFunc1606(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite solution with constraint handling
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible,:);
        elite = elite(elite_idx,:);
    else
        [~, elite_idx] = min(popfits + 1e6*abs(cons));
        elite = popdecs(elite_idx,:);
    end
    
    % 2. Compute normalized ranks for fitness and constraints
    [~, fit_rank_order] = sort(popfits);
    fit_ranks = zeros(NP,1);
    fit_ranks(fit_rank_order) = (1:NP)'/NP;
    
    [~, cons_rank_order] = sort(cons);
    cons_ranks = zeros(NP,1);
    cons_ranks(cons_rank_order) = (1:NP)'/NP;
    
    % 3. Adaptive scaling factors
    F = 0.5 * (1 + fit_ranks);
    G = 0.3 * (1 - cons_ranks);
    H = 0.2 * (cons ./ max(abs(cons)));
    H(isnan(H)) = 0;
    
    % 4. Direction-guided mutation with 6 random vectors
    idx = randperm(NP, 6*NP);
    r1 = reshape(idx(1:NP), [], 1);
    r2 = reshape(idx(NP+1:2*NP), [], 1);
    r3 = reshape(idx(2*NP+1:3*NP), [], 1);
    r4 = reshape(idx(3*NP+1:4*NP), [], 1);
    r5 = reshape(idx(4*NP+1:5*NP), [], 1);
    r6 = reshape(idx(5*NP+1:6*NP), [], 1);
    
    mutation = elite(ones(NP,1), :) + ...
               F.*(popdecs(r1,:) - popdecs(r2,:)) + ...
               G.*(popdecs(r3,:) - popdecs(r4,:)) + ...
               H.*(popdecs(r5,:) - popdecs(r6,:));
    
    % 5. Dynamic crossover rate with opposition-based learning
    CR = 0.9 - 0.5 * (fit_ranks + cons_ranks)/2;
    
    % 6. Crossover with opposition-based components
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    % Opposition-based components
    oppo_pop = lb + ub - popdecs;
    oppo_mask = rand(NP,D) < 0.1;
    
    offspring = popdecs .* (~mask) + mutation .* mask;
    offspring = offspring .* (~oppo_mask) + oppo_pop .* oppo_mask;
    
    % 7. Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring(below) = lb_rep(below) + rand(NP,D) .* (ub_rep(below) - lb_rep(below));
    offspring(above) = lb_rep(above) + rand(NP,D) .* (ub_rep(above) - lb_rep(above));
    
    % 8. Final boundary check
    offspring = min(max(offspring, lb_rep), ub_rep);
end