% MATLAB Code
function [offspring] = updateFunc1594(popdecs, popfits, cons)
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
    
    % 2. Rank-based scaling with constraint awareness
    [~, sorted_idx] = sort(popfits);
    [~, ranks] = sort(sorted_idx);
    max_cons = max(cons);
    F_base = 0.5 + 0.3 * tanh(1 - ranks/NP);
    F = F_base .* (1 - abs(cons)/(max_cons + 1));
    
    % 3. Generate random indices ensuring distinct vectors
    rand_idx1 = randi(NP, NP, 1);
    rand_idx2 = randi(NP, NP, 1);
    same_idx = rand_idx1 == rand_idx2;
    rand_idx2(same_idx) = mod(rand_idx2(same_idx) + randi(NP-1, sum(same_idx), 1), NP) + 1;
    
    % 4. Generate mutation vectors with elite guidance
    diff_vectors = popdecs(rand_idx1,:) - popdecs(rand_idx2,:);
    mutation = elite + F.*diff_vectors;
    
    % 5. Boundary handling with reflection
    lb_mask = mutation < lb;
    ub_mask = mutation > ub;
    mutation(lb_mask) = 2*lb(lb_mask) - mutation(lb_mask);
    mutation(ub_mask) = 2*ub(ub_mask) - mutation(ub_mask);
    
    % 6. Adaptive crossover based on constraint violation
    CR = 0.9 - 0.5*(cons - min(cons))./(max_cons - min(cons) + eps);
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    offspring = popdecs .* (~mask) + mutation .* mask;
    
    % 7. Final boundary enforcement
    offspring = min(max(offspring, lb), ub);
end