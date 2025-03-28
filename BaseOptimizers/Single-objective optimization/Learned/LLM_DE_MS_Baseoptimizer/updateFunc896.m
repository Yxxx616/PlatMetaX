% MATLAB Code
function [offspring] = updateFunc896(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Fitness-based weights with improved robustness
    median_fit = median(popfits);
    weights = exp(-abs(popfits - median_fit) / (std(popfits) + eps);
    weights = weights / sum(weights);
    
    % Fitness-weighted direction vector
    median_pop = median(popdecs, 1);
    dw = sum(bsxfun(@times, popdecs - median_pop, weights), 1);
    
    % Constraint-aware perturbation
    max_cons = max(abs(cons));
    alpha = 1 - tanh(abs(cons) / (max_cons + eps));
    beta = 0.2 * (1 - alpha);
    perturbation = bsxfun(@times, tanh(abs(cons)), sign(cons)) .* randn(NP, D);
    
    % Initialize offspring
    offspring = popdecs;
    F = 0.5;
    
    for i = 1:NP
        % Tournament selection for parents
        candidates = setdiff(1:NP, i);
        [~, idx] = sort(popfits(candidates));
        tournament_size = max(3, round(NP/4));
        selected = candidates(idx(1:tournament_size));
        r = selected(randperm(tournament_size, 3));
        r1 = r(1); r2 = r(2); r3 = r(3);
        
        % Composite mutation
        term1 = alpha(i) * dw;
        term2 = (1-alpha(i)) * (popdecs(r2,:) - popdecs(r3,:));
        mutant = popdecs(r1,:) + F * (term1 + term2) + beta(i) * perturbation(i,:);
        
        % Adaptive crossover rate
        CR = 0.9 * (1 - sqrt(i/NP)) * (0.9 + 0.2*rand());
        j_rand = randi(D);
        mask = rand(1,D) < CR;
        mask(j_rand) = true;
        
        % Create trial vector
        offspring(i,mask) = mutant(mask);
    end
    
    % Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Fitness-based reflection factor
    norm_fit = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    reflect_factor = 0.2 + 0.6 * (1 - norm_fit);
    reflect_factor = repmat(reflect_factor, 1, D);
    
    % Handle boundaries
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring(below) = lb_rep(below) + reflect_factor(below) .* (lb_rep(below) - offspring(below));
    offspring(above) = ub_rep(above) - reflect_factor(above) .* (offspring(above) - ub_rep(above));
    
    % Final clamping with small perturbation
    offspring = max(min(offspring, ub_rep), lb_rep);
    perturbation = 0.005 * (ub - lb) .* randn(NP, D) .* (1 - norm_fit);
    offspring = offspring + perturbation;
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Elite preservation with reduced mutation
    [~, sorted_idx] = sort(popfits);
    elite_size = max(1, round(0.15*NP));
    elite = popdecs(sorted_idx(1:elite_size),:);
    elite_mutation = 0.005 * (ub - lb) .* randn(elite_size, D);
    offspring(sorted_idx(1:elite_size),:) = elite + elite_mutation;
    offspring = max(min(offspring, ub_rep), lb_rep);
end