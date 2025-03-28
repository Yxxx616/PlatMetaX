% MATLAB Code
function [offspring] = updateFunc591(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints
    cons_pos = max(0, cons);
    max_cons = max(cons_pos) + eps;
    cons_norm = cons_pos ./ max_cons;
    
    % Elite selection with feasibility rules
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons_pos);
        elite = popdecs(elite_idx, :);
    end
    
    % Generate 4 distinct random indices for each individual
    idx = 1:NP;
    r1 = zeros(NP, 1); r2 = zeros(NP, 1); r3 = zeros(NP, 1); r4 = zeros(NP, 1);
    for i = 1:NP
        available = setdiff(idx, i);
        r1(i) = available(randi(NP-1));
        remaining = setdiff(available, r1(i));
        r2(i) = remaining(randi(NP-2));
        remaining = setdiff(remaining, r2(i));
        r3(i) = remaining(randi(NP-3));
        remaining = setdiff(remaining, r3(i));
        r4(i) = remaining(randi(NP-4));
    end
    
    % Adaptive scaling factors based on constraints
    F = 0.4 + 0.4 * (1 - cons_norm);
    
    % Mutation with elite guidance and dual differences
    elite_rep = repmat(elite, NP, 1);
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    diff2 = popdecs(r3,:) - popdecs(r4,:);
    mutant = elite_rep + F .* diff1 + 0.5 * F .* diff2;
    
    % Constraint-aware crossover
    CR = 0.2 + 0.6 * (1 - cons_norm);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1,D)) | repmat((1:D) == j_rand, 1, D);
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling - reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Final bounds check
    offspring = min(max(offspring, lb_rep), ub_rep);
end