% MATLAB Code
function [offspring] = updateFunc1576(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Constraint-aware fitness adjustment
    beta = 0.5;
    adj_fits = popfits + beta * max(0, cons);
    
    % 2. Sort population by adjusted fitness
    [~, sorted_idx] = sort(adj_fits);
    ranks = zeros(NP,1);
    ranks(sorted_idx) = (1:NP)';
    
    % 3. Get best individual
    x_best = popdecs(sorted_idx(1), :);
    
    % 4. Generate random indices (vectorized)
    rand_idx = zeros(NP, 3);
    for i = 1:NP
        avail = setdiff(1:NP, i);
        perm = avail(randperm(length(avail), 3));
        rand_idx(i,:) = perm;
    end
    r1 = rand_idx(:,1); r2 = rand_idx(:,2); r3 = rand_idx(:,3);
    
    % 5. Adaptive scaling factors
    F = 0.5 * (1 + ranks/NP);
    
    % 6. Directional mutation
    offspring = popdecs(r1,:) + ...
                F.*(repmat(x_best, NP, 1) - popdecs) + ...
                F.*(popdecs(r2,:) - popdecs(r3,:));
    
    % 7. Adaptive crossover
    CR = 0.1 + 0.7*(1 - ranks/NP);
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    offspring = popdecs .* (~mask) + offspring .* mask;
    
    % 8. Boundary handling with reflection and random perturbation
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    
    % Reflection with random perturbation
    offspring(lb_mask) = lb(lb_mask) + abs(offspring(lb_mask) - lb(lb_mask)).*rand(sum(lb_mask(:)),1);
    offspring(ub_mask) = ub(ub_mask) - abs(offspring(ub_mask) - ub(ub_mask)).*rand(sum(ub_mask(:)),1);
    
    % Final boundary enforcement
    offspring = min(max(offspring, lb), ub);
end