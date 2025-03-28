% MATLAB Code
function [offspring] = updateFunc861(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Elite identification (top 10%)
    [~, sorted_idx] = sort(popfits);
    elite_num = max(1, round(0.1*NP));
    elite = popdecs(sorted_idx(1:elite_num), :);
    
    % Best solution selection
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        best = popdecs(feasible(best_idx), :);
    else
        [~, best_idx] = min(cons);
        best = popdecs(best_idx, :);
    end
    
    % Adaptive scaling factor
    mean_c = mean(cons);
    std_c = std(cons) + eps;
    F = 0.5 + 0.3 * tanh((cons - mean_c) ./ std_c);
    
    % Mutation with random pairing
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    for i = 1:NP
        while r1(i) == i || r2(i) == i || r1(i) == r2(i)
            r1(i) = randi(NP);
            r2(i) = randi(NP);
        end
    end
    
    best_diff = best - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    mutant = popdecs + F.*best_diff + (1-F).*rand_diff;
    
    % Rank-based crossover
    [~, rank] = sort(popfits);
    CR = 0.9 - 0.4 * (rank-1)/NP;
    
    % Binomial crossover
    mask = rand(NP, D) < CR(:, ones(1, D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Elite-guided boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    for i = 1:NP
        elite_idx = randi(elite_num);
        if any(below_lb(i,:)) || any(above_ub(i,:))
            delta = rand(1,D);
            offspring(i,:) = elite(elite_idx,:) + delta.*(offspring(i,:) - elite(elite_idx,:));
        end
    end
    
    % Final boundary check
    offspring = max(min(offspring, ub_rep), lb_rep);
end