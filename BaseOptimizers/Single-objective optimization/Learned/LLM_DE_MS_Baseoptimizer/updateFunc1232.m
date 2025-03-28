% MATLAB Code
function [offspring] = updateFunc1232(popdecs, popfits, cons)
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
        available = setdiff(idx, [i, elite_idx]);
        R(i,:) = available(randperm(length(available), 4));
    end
    
    % 3. Compute fitness weights for difference vectors
    f_normalized = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    w = exp(-f_normalized(R(:,1:2))) ./ (exp(-f_normalized(R(:,1:2))) + exp(-f_normalized(R(:,3:4))) + eps);
    
    % 4. Compute directions
    elite_dir = x_elite - popdecs;
    diff_dir = w(:,1).*(popdecs(R(:,1),:) - popdecs(R(:,2),:)) + ...
               w(:,2).*(popdecs(R(:,3),:) - popdecs(R(:,4),:));
    
    % 5. Constraint-aware scaling factor
    mean_c = mean(cons);
    std_c = std(cons) + eps;
    F = 0.5 * (1 + tanh((cons - mean_c)/std_c));
    F = F(:, ones(1, D));
    
    % 6. Adaptive balance between elite and difference directions
    mean_f = mean(popfits);
    std_f = std(popfits) + eps;
    alpha = 1 ./ (1 + exp(-5*(popfits - mean_f)/std_f));
    alpha = alpha(:, ones(1, D));
    
    % 7. Mutation with adaptive components
    mutants = popdecs + F .* (alpha.*elite_dir + (1-alpha).*diff_dir);
    
    % 8. Rank-based adaptive crossover
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    CR = 0.9 * (1 - ranks/NP);
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