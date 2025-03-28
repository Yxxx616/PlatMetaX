% MATLAB Code
function [offspring] = updateFunc894(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Fitness-based weights calculation
    mean_fit = mean(popfits);
    weights = 1 ./ (1 + abs(popfits - mean_fit));
    weights = weights / sum(weights);
    
    % Fitness-weighted direction vector
    mean_pop = mean(popdecs, 1);
    dw = sum(bsxfun(@times, popdecs - mean_pop, weights), 1);
    
    % Constraint-aware scaling factors
    max_cons = max(abs(cons));
    alpha = 1 - tanh(abs(cons) / (max_cons + eps));
    
    % Initialize offspring
    offspring = popdecs;
    F = 0.5;
    beta = 0.2;
    
    for i = 1:NP
        % Select three distinct random indices
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 3));
        r1 = idx(1); r2 = idx(2); r3 = idx(3);
        
        % Composite mutation
        term1 = alpha(i) * dw;
        term2 = (1-alpha(i)) * (popdecs(r2,:) - popdecs(r3,:));
        constraint_term = beta * sign(cons(i)) * abs(cons(i))^1.5 .* randn(1,D);
        mutant = popdecs(r1,:) + F * (term1 + term2) + constraint_term;
        
        % Adaptive crossover rate
        CR = 0.9 * (1 - (i/NP)^1.5);
        j_rand = randi(D);
        mask = rand(1,D) < CR;
        mask(j_rand) = true;
        
        % Create trial vector
        offspring(i,mask) = mutant(mask);
    end
    
    % Boundary handling with fitness-based reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Normalized fitness for reflection
    norm_fit = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    reflect_factor = 0.2 + 0.6 * norm_fit;
    reflect_factor = repmat(reflect_factor, 1, D);
    
    % Handle boundaries
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring(below) = lb_rep(below) + reflect_factor(below) .* (lb_rep(below) - offspring(below));
    offspring(above) = ub_rep(above) - reflect_factor(above) .* (offspring(above) - ub_rep(above));
    
    % Final clamping with small perturbation
    offspring = max(min(offspring, ub_rep), lb_rep);
    perturbation = 0.02 * (ub - lb) .* randn(NP, D) .* (1 - norm_fit);
    offspring = offspring + perturbation;
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Preserve top 5% solutions
    [~, sorted_idx] = sort(popfits);
    elite_size = max(1, round(0.05*NP));
    offspring(sorted_idx(1:elite_size),:) = popdecs(sorted_idx(1:elite_size),:);
end