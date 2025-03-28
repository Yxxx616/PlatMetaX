% MATLAB Code
function [offspring] = updateFunc157(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Calculate ranks and normalized constraint violations
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_idx) = (1:NP)'/NP;
    norm_cons = abs(cons) / (max(abs(cons)) + eps);
    
    % 2. Adaptive parameters
    F = 0.4 + 0.3 * tanh(1 - norm_cons) .* (1 - ranks);
    CR = 0.7 + 0.2 * (1 - norm_cons);
    
    % 3. Elite selection (top 20%)
    [~, sorted_idx] = sort(popfits);
    elite_idx = sorted_idx(1:ceil(0.2*NP));
    
    % 4. Generate random indices (4 distinct indices per individual)
    rand_idx = zeros(NP, 4);
    for i = 1:NP
        available = setdiff(1:NP, i);
        rand_idx(i,:) = available(randperm(length(available), 4));
    end
    
    % 5. Select random elite for each individual
    elite_rand_idx = elite_idx(randi(length(elite_idx), NP, 1));
    
    % 6. Direction vectors with adaptive exploration-exploitation
    explore_mask = rand(NP,1) < 0.5;
    d = zeros(NP, D);
    
    % Exploration direction (DE/rand/2)
    r1 = rand_idx(:,1); r2 = rand_idx(:,2);
    r3 = rand_idx(:,3); r4 = rand_idx(:,4);
    d(explore_mask,:) = popdecs(r1(explore_mask),:) - popdecs(r2(explore_mask),:) + ...
                        popdecs(r3(explore_mask),:) - popdecs(r4(explore_mask),:);
    
    % Exploitation direction (DE/current-to-elite/1)
    d(~explore_mask,:) = popdecs(elite_rand_idx(~explore_mask),:) - popdecs(~explore_mask,:) + ...
                         F(~explore_mask).*(popdecs(r1(~explore_mask),:) - popdecs(r2(~explore_mask),:));
    
    % 7. Mutation
    v = popdecs + F.*d;
    
    % 8. Crossover with adaptive CR
    CR_mat = repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR_mat;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % 9. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection for boundary violations
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Final projection for any remaining violations
    offspring = max(min(offspring, ub_rep), lb_rep);
end