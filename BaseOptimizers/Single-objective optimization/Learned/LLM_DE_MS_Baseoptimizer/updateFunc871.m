% MATLAB Code
function [offspring] = updateFunc871(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Feasibility information
    feasible = cons <= 0;
    num_feasible = sum(feasible);
    
    % Best individual selection
    if num_feasible > 0
        [~, best_idx] = min(popfits(feasible));
        best = popdecs(feasible(best_idx), :);
    else
        [~, best_idx] = min(cons);
        best = popdecs(best_idx, :);
    end
    
    % Adaptive parameters
    sigma_c = std(cons) + eps;
    mu_c = mean(cons);
    F = 0.5 + 0.3 * tanh(cons/(sigma_c + eps));
    beta = 0.5 * (1 - tanh(cons/(max(cons) + eps)));
    
    % Selection probabilities
    p = 1./(1 + exp(5*cons/(sigma_c + eps)));
    p = p/sum(p);
    
    % Mutation
    mutant = zeros(NP, D);
    for i = 1:NP
        % Select base vector
        if rand() < 0.7 && num_feasible > 0
            base_idx = randsample(find(feasible), 1, true, p(feasible));
        else
            base_idx = randsample(NP, 1, true, p);
        end
        base = popdecs(base_idx, :);
        
        % Select difference vectors
        candidates = setdiff(1:NP, base_idx);
        selected = candidates(randperm(length(candidates), 2));
        vec2 = popdecs(selected(1), :);
        vec3 = popdecs(selected(2), :);
        
        % Mutation
        mutant(i,:) = base + F(i)*(vec2 - vec3) + beta(i)*(best - base);
        
        % Constraint-driven diversity
        if cons(i) > mu_c + sigma_c
            mutant(i,:) = mutant(i,:) + sigma_c*randn(1,D);
        end
    end
    
    % Dynamic crossover
    CR = 0.9 - 0.4*(cons/(max(cons) + eps));
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
    
    % Extreme constraint handling
    extreme_cons = cons > (mu_c + 2*sigma_c);
    if any(extreme_cons)
        dims = randi(D, sum(extreme_cons), 1);
        idx = find(extreme_cons);
        for i = 1:length(idx)
            offspring(idx(i), dims(i)) = lb(dims(i)) + rand()*(ub(dims(i))-lb(dims(i)));
        end
    end
end