% MATLAB Code
function [offspring] = updateFunc887(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Feasibility-aware weighting
    feasible = cons <= 0;
    alpha = 1.0; % Penalty factor
    weighted_fits = popfits;
    weighted_fits(~feasible) = weighted_fits(~feasible) + alpha * abs(cons(~feasible));
    
    % Sort by weighted fitness
    [~, sorted_idx] = sort(weighted_fits);
    elite_size = max(3, round(NP/5));
    
    % Directional vector computation
    K = min(5, elite_size);
    top_K = popdecs(sorted_idx(1:K),:);
    bottom_K = popdecs(sorted_idx(end-K+1:end),:);
    directional_vec = mean(top_K - bottom_K, 1);
    
    % Initialize offspring
    offspring = zeros(NP, D);
    p_elite = 0.7;
    
    for i = 1:NP
        % Base vector selection
        if rand() < p_elite
            base = popdecs(sorted_idx(1),:);
        else
            base = popdecs(randi(NP),:);
        end
        
        % Adaptive parameters
        F = 0.5 + 0.3 * cos(pi * i/NP);
        CR = 0.9 - 0.5 * i/NP;
        
        % Select two distinct random vectors
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 2));
        r1 = idx(1); r2 = idx(2);
        
        % Directional mutation
        mutant = base + F * directional_vec + F * (popdecs(r1,:) - popdecs(r2,:));
        
        % Binomial crossover
        j_rand = randi(D);
        mask = rand(1,D) < CR;
        mask(j_rand) = true;
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutant(mask);
    end
    
    % Boundary handling - mirroring
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    offspring(below) = 2*lb_rep(below) - offspring(below);
    offspring(above) = 2*ub_rep(above) - offspring(above);
    
    % Final clamping
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Preserve best solution
    offspring(sorted_idx(1),:) = popdecs(sorted_idx(1),:);
end