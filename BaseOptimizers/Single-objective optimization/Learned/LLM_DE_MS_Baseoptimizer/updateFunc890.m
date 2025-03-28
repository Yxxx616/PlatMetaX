% MATLAB Code
function [offspring] = updateFunc890(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Enhanced feasibility-aware ranking with dynamic penalty
    alpha = 2.0; % Increased penalty factor
    feasible = cons <= 0;
    rank_scores = popfits;
    rank_scores(~feasible) = rank_scores(~feasible) + alpha * (abs(cons(~feasible)).^1.3);
    
    % Sort population by rank scores
    [~, sorted_idx] = sort(rank_scores);
    elite_size = max(5, round(NP/4)); % Adjusted elite pool
    K = min(8, elite_size); % More direction vectors
    
    % Compute elite-guided direction vectors
    top_K = popdecs(sorted_idx(1:K),:);
    bottom_K = popdecs(sorted_idx(end-K+1:end),:);
    directional_vec = mean(top_K - bottom_K, 1);
    
    % Initialize offspring
    offspring = zeros(NP, D);
    p_elite = 0.85; % Higher elite probability
    beta = 0.3; % Adjusted constraint influence
    
    for i = 1:NP
        % Base vector selection with higher elite probability
        if rand() < p_elite
            base = popdecs(sorted_idx(randi(ceil(elite_size/2))),:); % Random from top half elite
        else
            base = popdecs(randi(NP),:);
        end
        
        % Adaptive parameters with stronger variation
        F = 0.6 + 0.2 * sin(pi * i/NP);
        CR = 0.95 - 0.4 * (i/NP)^2;
        
        % Select two distinct random vectors
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 2));
        r1 = idx(1); r2 = idx(2);
        
        % Enhanced constraint-aware perturbation
        r = randn(1,D); % Gaussian noise
        constraint_term = beta * sign(cons(i)) * abs(cons(i))^1.2 * r;
        
        % Improved mutation with directional guidance
        mutant = base + F * directional_vec + F * (popdecs(r1,:) - popdecs(r2,:)) + constraint_term;
        
        % Dynamic binomial crossover
        j_rand = randi(D);
        mask = rand(1,D) < CR;
        mask(j_rand) = true;
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutant(mask);
    end
    
    % Advanced boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    % Adaptive bounce-back with fitness-based scaling
    rank_ratio = (rank_scores - min(rank_scores)) ./ (max(rank_scores) - min(rank_scores) + eps);
    scale_factors = 0.25 + 0.5 * rank_ratio;
    scale_factors = repmat(scale_factors, 1, D);
    
    offspring(below) = lb_rep(below) + scale_factors(below) .* (popdecs(below) - lb_rep(below));
    offspring(above) = ub_rep(above) - scale_factors(above) .* (ub_rep(above) - popdecs(above));
    
    % Final clamping with small random perturbation
    offspring = max(min(offspring, ub_rep), lb_rep);
    offspring = offspring + 0.008 * (ub - lb) .* randn(NP, D);
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Preserve top 4 solutions
    offspring(sorted_idx(1:4),:) = popdecs(sorted_idx(1:4),:);
end