% MATLAB Code
function [offspring] = updateFunc874(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Feasibility information
    feasible = cons <= 0;
    rho = sum(feasible)/NP;
    
    % Normalized constraint violation
    min_cons = min(cons);
    max_cons = max(cons);
    s = (cons - min_cons)/(max_cons - min_cons + eps);
    
    % Adaptive parameters
    F = 0.5 + 0.3 * tanh(1 - s);
    alpha = 0.5 * (1 + tanh(s - 0.5));
    sigma = 0.2 * (1 - exp(-5*s)) .* (ub - lb);
    beta = 0.5;
    
    % Best individual selection
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        best = popdecs(feasible(best_idx), :);
    else
        [~, best_idx] = min(cons);
        best = popdecs(best_idx, :);
    end
    
    % Mutation
    mutant = zeros(NP, D);
    for i = 1:NP
        % Base vector selection
        if rho > 0.5
            base = best;
        else
            candidates = setdiff(1:NP, i);
            base_idx = candidates(randi(length(candidates)));
            base = popdecs(base_idx, :);
        end
        
        % Difference vectors
        candidates = setdiff(1:NP, i);
        selected = candidates(randperm(length(candidates), 2));
        vec2 = popdecs(selected(1), :);
        vec3 = popdecs(selected(2), :);
        
        % Constraint-aware mutation
        mutant(i,:) = base + (F(i)*(vec2 - vec3) + alpha(i)*sigma(i).*randn(1,D)) .* (1 + beta*s(i));
    end
    
    % Dynamic crossover
    CR = 0.9 - 0.4*s;
    mask = rand(NP, D) < CR(:, ones(1, D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Final clamping
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Constraint-based dimension reset
    [~, sorted_cons] = sort(cons, 'descend');
    num_reset = ceil(0.1*NP);
    for i = 1:num_reset
        idx = sorted_cons(i);
        dim = randi(D);
        offspring(idx, dim) = lb(dim) + rand()*(ub(dim)-lb(dim));
    end
end