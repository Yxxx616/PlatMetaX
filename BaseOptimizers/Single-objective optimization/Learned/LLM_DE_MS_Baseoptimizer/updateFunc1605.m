% MATLAB Code
function [offspring] = updateFunc1605(popdecs, popfits, cons)
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
        [~, elite_idx] = min(popfits + 1e6*max(0, cons));
        elite = popdecs(elite_idx,:);
    end
    
    % 2. Compute ranks for fitness and constraints
    [~, fit_rank_order] = sort(popfits);
    fit_ranks = zeros(NP,1);
    fit_ranks(fit_rank_order) = 1:NP;
    
    [~, cons_rank_order] = sort(cons);
    cons_ranks = zeros(NP,1);
    cons_ranks(cons_rank_order) = 1:NP;
    
    % 3. Adaptive scaling factors
    F = 0.4 + 0.3*(fit_ranks/NP) - 0.2*(cons_ranks/NP);
    G = 0.2*(1 - (fit_ranks + cons_ranks)/(2*NP));
    
    % 4. Direction-guided mutation with 4 random vectors
    idx = randperm(NP, 4*NP);
    r1 = reshape(idx(1:NP), [], 1);
    r2 = reshape(idx(NP+1:2*NP), [], 1);
    r3 = reshape(idx(2*NP+1:3*NP), [], 1);
    r4 = reshape(idx(3*NP+1:4*NP), [], 1);
    
    mutation = elite(ones(NP,1), :) + ...
               F.*(popdecs(r1,:) - popdecs(r2,:)) + ...
               G.*(popdecs(r3,:) - popdecs(r4,:));
    
    % 5. Dynamic crossover rate
    CR = 0.85 - 0.4 * (fit_ranks + cons_ranks)/(2*NP);
    
    % 6. Crossover
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs .* (~mask) + mutation .* mask;
    
    % 7. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring(below) = 2*lb_rep(below) - offspring(below);
    offspring(above) = 2*ub_rep(above) - offspring(above);
    
    % 8. Small perturbation (5% probability)
    perturb_mask = rand(NP,D) < 0.05;
    perturb_amount = randn(NP,D) .* (ub_rep - lb_rep) * 0.01;
    offspring = offspring + perturb_mask .* perturb_amount;
    
    % Final boundary check
    offspring = min(max(offspring, lb_rep), ub_rep);
end