% MATLAB Code
function [offspring] = updateFunc895(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Fitness-based weights calculation (improved normalization)
    median_fit = median(popfits);
    weights = exp(-abs(popfits - median_fit) / (std(popfits) + eps));
    weights = weights / sum(weights);
    
    % Fitness-weighted direction vector
    median_pop = median(popdecs, 1);
    dw = sum(bsxfun(@times, popdecs - median_pop, weights), 1);
    
    % Constraint-aware perturbation
    max_cons = max(abs(cons));
    alpha = 1 - tanh(abs(cons) / (max_cons + eps));
    beta = 0.3 * (1 - alpha);
    perturbation = bsxfun(@times, tanh(abs(cons)), sign(cons)) .* randn(NP, D);
    
    % Initialize offspring
    offspring = popdecs;
    F = 0.6;
    
    for i = 1:NP
        % Select three distinct random indices using tournament selection
        candidates = setdiff(1:NP, i);
        [~, idx] = sort(popfits(candidates));
        r1 = candidates(idx(1));
        r2 = candidates(idx(1+randi(round(NP/3)));
        r3 = candidates(idx(1+randi(round(NP/3))));
        
        % Composite mutation with improved exploration
        term1 = alpha(i) * dw;
        term2 = (1-alpha(i)) * (popdecs(r2,:) - popdecs(r3,:));
        mutant = popdecs(r1,:) + F * (term1 + term2) + beta(i) * perturbation(i,:);
        
        % Adaptive crossover rate with jitter
        CR = 0.9 * (1 - sqrt(i/NP)) * (0.95 + 0.1*rand());
        j_rand = randi(D);
        mask = rand(1,D) < CR;
        mask(j_rand) = true;
        
        % Create trial vector
        offspring(i,mask) = mutant(mask);
    end
    
    % Boundary handling with fitness-based reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Improved reflection factor
    norm_fit = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    reflect_factor = 0.1 + 0.7 * (1 - norm_fit);  % Better solutions get smaller reflection
    reflect_factor = repmat(reflect_factor, 1, D);
    
    % Handle boundaries with reflection
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring(below) = lb_rep(below) + reflect_factor(below) .* (lb_rep(below) - offspring(below));
    offspring(above) = ub_rep(above) - reflect_factor(above) .* (offspring(above) - ub_rep(above));
    
    % Final clamping with adaptive perturbation
    offspring = max(min(offspring, ub_rep), lb_rep);
    perturbation = 0.01 * (ub - lb) .* randn(NP, D) .* (1 - norm_fit);
    offspring = offspring + perturbation;
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Preserve top 10% solutions with small mutation
    [~, sorted_idx] = sort(popfits);
    elite_size = max(1, round(0.1*NP));
    elite = popdecs(sorted_idx(1:elite_size),:);
    elite = elite + 0.01 * (ub - lb) .* randn(elite_size, D);
    offspring(sorted_idx(1:elite_size),:) = max(min(elite, ub_rep(sorted_idx(1:elite_size),:)), ...
                                          lb_rep(sorted_idx(1:elite_size),:));
end