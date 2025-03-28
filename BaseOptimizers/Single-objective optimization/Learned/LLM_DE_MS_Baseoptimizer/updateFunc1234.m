% MATLAB Code
function [offspring] = updateFunc1234(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite considering both fitness and constraints
    penalty = popfits + 1000 * max(0, cons);
    [~, elite_idx] = min(penalty);
    x_elite = popdecs(elite_idx, :);
    
    % 2. Generate random indices matrix (vectorized)
    idx = 1:NP;
    R = zeros(NP, 4);
    for i = 1:NP
        available = idx(~ismember(idx, [i, elite_idx]));
        R(i,:) = available(randperm(length(available), 4));
    end
    
    % 3. Compute fitness weights for difference vectors
    f_min = min(popfits);
    f_max = max(popfits);
    f_normalized = (popfits - f_min) / (f_max - f_min + eps);
    exp_f = exp(-f_normalized(R));
    w = exp_f(:,1:2) ./ (exp_f(:,1:2) + exp_f(:,3:4) + eps);
    
    % 4. Compute directions
    elite_dir = x_elite - popdecs;
    diff_dir = w(:,1).*(popdecs(R(:,1),:) - popdecs(R(:,2),:)) + ...
               w(:,2).*(popdecs(R(:,3),:) - popdecs(R(:,4),:));
    
    % 5. Adaptive balance between elite and difference directions
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    alpha = 0.3 + 0.5 * (ranks / NP);
    alpha = alpha(:, ones(1, D));
    
    % 6. Constraint-driven perturbation
    beta = 0.1 * tanh(abs(cons));
    epsilon = 0.01 * randn(NP, D);
    cons_perturb = beta .* sign(cons) .* epsilon;
    
    % 7. Mutation with adaptive components
    F = 0.6;
    mutants = popdecs + F .* (alpha.*elite_dir + (1-alpha).*diff_dir) + cons_perturb;
    
    % 8. Rank-based adaptive crossover
    CR = 0.6 + 0.3 * (1 - ranks/NP);
    CR = CR(:, ones(1, D));
    
    % Perform crossover
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    
    % Final bounds check
    offspring = min(max(offspring, lb), ub);
end