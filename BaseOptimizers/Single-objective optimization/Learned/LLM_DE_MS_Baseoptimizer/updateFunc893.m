% MATLAB Code
function [offspring] = updateFunc893(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Enhanced constraint handling with adaptive penalty
    feasible = cons <= 0;
    worst_fit = max(popfits);
    best_fit = min(popfits);
    penalty = 2.5 * abs(cons).^1.5;
    rank_scores = popfits + (~feasible) .* penalty;
    
    % Sort population by rank scores
    [~, sorted_idx] = sort(rank_scores);
    elite_size = max(4, round(NP/3));
    K = min(6, elite_size);
    
    % Compute elite-guided direction vectors
    top_K = popdecs(sorted_idx(1:K),:);
    bottom_K = popdecs(sorted_idx(end-K+1:end),:);
    directional_vec = mean(top_K - bottom_K, 1);
    
    % Initialize offspring
    offspring = zeros(NP, D);
    max_cons = max(abs(cons));
    
    for i = 1:NP
        % Adaptive elite probability based on constraints
        p_elite = 0.9 - 0.5 * (abs(cons(i)) / (max_cons + eps));
        
        % Base vector selection
        if rand() < p_elite
            base_idx = sorted_idx(randi(elite_size));
        else
            base_idx = randi(NP);
        end
        base = popdecs(base_idx,:);
        base_fit = popfits(base_idx);
        
        % Adaptive parameters
        F_base = 0.5 + 0.3 * cos(pi * i/NP);
        F_i = F_base * (1 + (base_fit - worst_fit)/(best_fit - worst_fit + eps));
        CR = 0.9 - 0.4 * (i/NP)^1.5;
        
        % Select two distinct random vectors
        candidates = setdiff(1:NP, [i, base_idx]);
        idx = candidates(randperm(length(candidates), 2));
        r1 = idx(1); r2 = idx(2);
        
        % Fitness-weighted difference vector
        fit_diff = 1 + abs(popfits(r1) - popfits(r2));
        delta_fit = (popdecs(r1,:) - popdecs(r2,:)) / fit_diff;
        
        % Constraint-aware perturbation
        beta = 0.35;
        constraint_term = beta * sign(cons(i)) * abs(cons(i))^1.5 .* randn(1,D);
        
        % Composite mutation
        mutant = base + F_i * directional_vec + F_i * delta_fit + constraint_term;
        
        % Dynamic crossover
        j_rand = randi(D);
        mask = rand(1,D) < CR;
        mask(j_rand) = true;
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutant(mask);
    end
    
    % Advanced boundary handling with fitness-based reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Normalized fitness for boundary handling
    norm_fit = (rank_scores - min(rank_scores)) ./ (max(rank_scores) - min(rank_scores) + eps);
    reflect_factor = 0.3 + 0.5 * norm_fit;
    reflect_factor = repmat(reflect_factor, 1, D);
    
    % Handle out-of-bounds solutions
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    offspring(below) = lb_rep(below) + reflect_factor(below) .* (lb_rep(below) - offspring(below));
    offspring(above) = ub_rep(above) - reflect_factor(above) .* (offspring(above) - ub_rep(above));
    
    % Final clamping with small adaptive perturbation
    offspring = max(min(offspring, ub_rep), lb_rep);
    perturbation = 0.01 * (ub - lb) .* (1 - norm_fit) .* randn(NP, D);
    offspring = offspring + perturbation;
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Preserve top 3 solutions
    offspring(sorted_idx(1:3),:) = popdecs(sorted_idx(1:3),:);
end