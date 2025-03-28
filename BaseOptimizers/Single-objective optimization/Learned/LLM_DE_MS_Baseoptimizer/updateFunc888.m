% MATLAB Code
function [offspring] = updateFunc888(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Feasibility-aware ranking
    alpha = 1.5; % Penalty factor
    feasible = cons <= 0;
    rank_scores = popfits;
    rank_scores(~feasible) = rank_scores(~feasible) + alpha * abs(cons(~feasible));
    
    % Sort population by rank scores
    [~, sorted_idx] = sort(rank_scores);
    elite_size = max(3, round(NP/4));
    K = min(5, elite_size);
    
    % Compute elite-guided direction vector
    top_K = popdecs(sorted_idx(1:K),:);
    bottom_K = popdecs(sorted_idx(end-K+1:end),:);
    directional_vec = mean(top_K - bottom_K, 1);
    
    % Initialize offspring
    offspring = zeros(NP, D);
    p_elite = 0.75;
    beta = 0.2; % Constraint influence factor
    
    for i = 1:NP
        % Base vector selection
        if rand() < p_elite
            base = popdecs(sorted_idx(1),:);
        else
            base = popdecs(randi(NP),:);
        end
        
        % Adaptive parameters
        F = 0.4 + 0.3 * sin(pi * i/NP);
        CR = 0.9 - 0.4 * (i/NP)^2;
        
        % Select two distinct random vectors
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 2));
        r1 = idx(1); r2 = idx(2);
        
        % Constraint-aware perturbation
        r = randn(1,D); % Gaussian noise
        constraint_term = beta * sign(cons(i)) * abs(cons(i)) * r;
        
        % Enhanced mutation
        mutant = base + F * directional_vec + F * (popdecs(r1,:) - popdecs(r2,:)) + constraint_term;
        
        % Binomial crossover
        j_rand = randi(D);
        mask = rand(1,D) < CR;
        mask(j_rand) = true;
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutant(mask);
    end
    
    % Boundary handling - bounce back
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    offspring(below) = lb_rep(below) + 0.5 * (popdecs(below) - lb_rep(below));
    offspring(above) = ub_rep(above) - 0.5 * (ub_rep(above) - popdecs(above));
    
    % Final clamping
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Preserve best solution
    [~, best_idx] = min(rank_scores);
    offspring(best_idx,:) = popdecs(best_idx,:);
end