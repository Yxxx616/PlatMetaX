% MATLAB Code
function [offspring] = updateFunc158(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Normalize constraint violations and calculate ranks
    norm_cons = abs(cons) / (max(abs(cons)) + eps);
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_idx) = (1:NP)'/NP;
    
    % 2. Adaptive parameters
    F = 0.5 * (1 + tanh(1 - norm_cons)) .* (1 - ranks);
    CR = 0.9 - 0.3 * norm_cons;
    
    % 3. Identify best individual
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx,:);
    
    % 4. Generate random indices (3 distinct indices per individual)
    rand_idx = zeros(NP, 3);
    for i = 1:NP
        available = setdiff(1:NP, i);
        rand_idx(i,:) = available(randperm(length(available), 3));
    end
    r1 = rand_idx(:,1); r2 = rand_idx(:,2); r3 = rand_idx(:,3);
    
    % 5. Hybrid mutation
    v = zeros(NP, D);
    use_rand = rand(NP,1) < 0.6;
    
    % DE/rand/1 mutation
    v(use_rand,:) = popdecs(r1(use_rand),:) + ...
                    F(use_rand).*(popdecs(r2(use_rand),:) - popdecs(r3(use_rand),:));
    
    % DE/current-to-best/1 mutation
    v(~use_rand,:) = popdecs(~use_rand,:) + ...
                     F(~use_rand).*(x_best - popdecs(~use_rand,:)) + ...
                     F(~use_rand).*(popdecs(r1(~use_rand),:) - popdecs(r2(~use_rand),:));
    
    % 6. Crossover with adaptive CR
    CR_mat = repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR_mat;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % 7. Constraint-aware boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Repair with midpoint between boundary and violated value
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = (offspring(below_lb) + lb_rep(below_lb)) / 2;
    offspring(above_ub) = (offspring(above_ub) + ub_rep(above_ub)) / 2;
    
    % Final projection for any remaining violations
    offspring = max(min(offspring, ub_rep), lb_rep);
end