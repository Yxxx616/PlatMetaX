% MATLAB Code
function [offspring] = updateFunc1593(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Elite selection with constraint handling
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, elite_idx] = min(popfits + 1e6*max(0, cons));
        elite = popdecs(elite_idx, :);
    end
    
    % 2. Rank-based scaling factors
    [~, sorted_idx] = sort(popfits);
    [~, ranks] = sort(sorted_idx);
    F = 0.5 + 0.3 * tanh(1 - ranks/NP);
    
    % 3. Constraint-aware adaptation
    min_cons = min(cons);
    max_cons = max(cons);
    alpha = 0.7 + 0.2 * (cons - min_cons)./(max_cons - min_cons + eps);
    
    % 4. Generate random indices
    rand_idx1 = randi(NP, NP, 1);
    rand_idx2 = randi(NP, NP, 1);
    same_idx = rand_idx1 == rand_idx2;
    rand_idx2(same_idx) = mod(rand_idx2(same_idx) + randi(NP-1, sum(same_idx), 1), NP) + 1;
    
    % 5. Generate direction vectors
    beta = 0.5 + 0.1 * randn(NP, 1);
    dir_vectors = popdecs(rand_idx1,:) - popdecs(rand_idx2,:) + ...
                 beta.*(elite - popdecs);
    
    % 6. Generate mutation vectors
    offspring = popdecs + F.*alpha.*dir_vectors;
    
    % 7. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    
    % 8. Final boundary enforcement
    offspring = min(max(offspring, lb), ub);
    
    % 9. Adaptive crossover based on constraint violation
    CR = 0.8 - 0.4*(cons - min_cons)./(max_cons - min_cons + eps);
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    offspring = popdecs .* (~mask) + offspring .* mask;
end