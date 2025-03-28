% MATLAB Code
function [offspring] = updateFunc98(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Normalize constraints
    c_max = max(abs(cons));
    norm_cons = abs(cons) / (c_max + eps);
    
    % Rank population (lower fitness is better)
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_order) = (1:NP)';
    norm_ranks = ranks / NP;
    
    % Combined weights (alpha=0.5 for balance)
    weights = 0.5 * norm_cons + 0.5 * (1 - norm_ranks);
    
    % Find best individual considering both fitness and constraints
    [~, best_idx] = min(popfits + norm_cons);
    x_best = popdecs(best_idx, :);
    
    % Parameters
    F = 0.7;
    
    % Tournament selection for r1, r2, r3
    r1 = zeros(NP,1); r2 = r1; r3 = r1;
    for i = 1:NP
        candidates = setdiff(1:NP, i);
        [~, idx] = max(weights(candidates));
        r1(i) = candidates(idx);
        candidates = setdiff(candidates, r1(i));
        [~, idx] = max(weights(candidates));
        r2(i) = candidates(idx);
        candidates = setdiff(candidates, r2(i));
        [~, idx] = max(weights(candidates));
        r3(i) = candidates(idx);
    end
    
    % Get selected vectors
    x_r1 = popdecs(r1, :);
    x_r2 = popdecs(r2, :);
    x_r3 = popdecs(r3, :);
    
    % Mutation
    diff1 = x_r1 - popdecs;
    diff2 = x_r2 - x_r3;
    v = x_best + F * diff1 + F * diff2;
    
    % Adaptive CR based on constraints
    CR = 0.9 * (1 - norm_cons) + 0.1;
    
    % Binomial crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = v(mask);
end