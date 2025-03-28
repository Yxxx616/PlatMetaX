% MATLAB Code
function [offspring] = updateFunc1239(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite considering both fitness and constraints
    penalty = popfits + 1000 * max(0, cons);
    [~, elite_idx] = min(penalty);
    x_elite = popdecs(elite_idx, :);
    
    % 2. Generate neighborhood indices (vectorized)
    idx = 1:NP;
    R = zeros(NP, 4);
    for i = 1:NP
        available = idx(~ismember(idx, [i, elite_idx]));
        R(i,:) = available(randperm(length(available), 4));
    end
    
    % 3. Compute fitness weights for neighbors
    f_min = min(popfits);
    f_max = max(popfits);
    f_normalized = (popfits - f_min) / (f_max - f_min + eps);
    exp_f = exp(-f_normalized(R));
    w = exp_f ./ (sum(exp_f, 2) + eps);
    
    % 4. Compute weighted difference vectors
    diff_vec = w(:,1).*(popdecs(R(:,1),:) - popdecs(R(:,2),:)) + ...
               w(:,2).*(popdecs(R(:,3),:) - popdecs(R(:,4),:));
    
    % 5. Rank-based adaptation
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    F = 0.5 * (1 + ranks/NP);
    F = F(:, ones(1, D));
    
    % 6. Constraint-aware perturbation
    beta = 0.1 * tanh(abs(cons));
    epsilon = 0.1 * randn(NP, D);
    cons_perturb = beta .* sign(cons) .* epsilon;
    
    % 7. Mutation with elite guidance
    elite_dir = (x_elite - popdecs) / 2;
    mutants = popdecs + F .* (elite_dir + diff_vec/2) + cons_perturb;
    
    % 8. Adaptive crossover
    CR = 0.9 - 0.5 * (ranks/NP);
    CR = CR(:, ones(1, D));
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 9. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    offspring = min(max(offspring, lb), ub);
end