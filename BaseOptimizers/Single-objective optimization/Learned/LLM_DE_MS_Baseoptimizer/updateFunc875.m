% MATLAB Code
function [offspring] = updateFunc875(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Feasibility information
    feasible = cons <= 0;
    rho = sum(feasible)/NP;
    
    % Rank-based scaling factors
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_idx) = 1:NP;
    F = 0.4 + 0.3 * (ranks/NP);
    
    % Normalized constraint violation
    min_cons = min(cons);
    max_cons = max(cons);
    s = (cons - min_cons)/(max_cons - min_cons + 1e-12);
    
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
        if rand() < rho
            base = best;
        else
            candidates = setdiff(1:NP, i);
            base_idx = candidates(randi(length(candidates)));
            base = popdecs(base_idx, :);
        end
        
        % Difference vectors
        candidates = setdiff(1:NP, [i, base_idx]);
        selected = candidates(randperm(length(candidates), 2));
        vec2 = popdecs(selected(1), :);
        vec3 = popdecs(selected(2), :);
        
        % Constraint direction factor
        if cons(i) > 0
            cons_sum = cons(i) + cons(selected(1)) + cons(selected(2));
            delta = cons(i)/cons_sum;
        else
            delta = 0;
        end
        
        % Adaptive step size
        sigma = 0.2 * (ub - lb) .* s(i);
        
        % Hybrid mutation
        mutant(i,:) = base + F(i)*(vec2 - vec3).*(1 - delta) + sigma.*randn(1,D).*delta;
    end
    
    % Adaptive crossover
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
    
    % Constraint-based dimension perturbation
    [~, sorted_cons] = sort(cons, 'descend');
    num_perturb = ceil(0.1*NP);
    perturb_idx = sorted_cons(1:num_perturb);
    for i = 1:num_perturb
        dim = randi(D);
        offspring(perturb_idx(i), dim) = lb(dim) + rand()*(ub(dim)-lb(dim));
    end
end