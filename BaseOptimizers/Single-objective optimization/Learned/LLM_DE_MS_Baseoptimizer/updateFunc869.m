% MATLAB Code
function [offspring] = updateFunc869(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Feasibility information
    feasible = cons <= 0;
    num_feasible = sum(feasible);
    
    % Combined ranking considering constraints
    alpha = 1e6; % Penalty factor for constraints
    combined_fits = popfits + alpha * max(0, cons);
    [~, sorted_idx] = sort(combined_fits);
    rank = zeros(NP,1);
    rank(sorted_idx) = 1:NP;
    
    % Best individual selection
    if num_feasible > 0
        [~, best_idx] = min(popfits(feasible));
        best = popdecs(feasible(best_idx), :);
    else
        [~, best_idx] = min(combined_fits);
        best = popdecs(best_idx, :);
    end
    
    % Adaptive scaling factors
    mean_c = mean(cons);
    std_c = std(cons) + eps;
    F = 0.5 * (1 + tanh((cons - mean_c)./std_c));
    F = min(max(F, 0.1), 0.9);
    
    % Constraint-aware weights
    beta = cons./(max(cons) + eps);
    
    % Rank-based crossover probability
    CR = 0.9 - 0.5 * (rank-1)/NP;
    
    % Mutation
    mutant = zeros(NP, D);
    for i = 1:NP
        % Select 4 distinct random indices
        candidates = setdiff(1:NP, i);
        selected = candidates(randperm(length(candidates), 4));
        r1 = selected(1); r2 = selected(2);
        r3 = selected(3); r4 = selected(4);
        
        % Hybrid mutation
        if num_feasible > 0
            mutant(i,:) = best + F(i)*(popdecs(r1,:)-popdecs(r2,:)) + ...
                         beta(i)*(popdecs(r3,:)-popdecs(r4,:));
        else
            % Fallback to rank-based mutation
            mutant(i,:) = popdecs(i,:) + F(i)*(best-popdecs(i,:)) + ...
                         F(i)*(popdecs(r1,:)-popdecs(r2,:));
        end
    end
    
    % Binomial crossover
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
    
    % Constraint-based diversity enhancement
    if max(cons) > min(cons)
        reset_prob = 0.1 * (cons - min(cons))/(max(cons) - min(cons) + eps);
        reset_mask = rand(NP,1) < reset_prob;
        if any(reset_mask)
            dims = randi(D, sum(reset_mask), 1);
            idx = find(reset_mask);
            for i = 1:length(idx)
                offspring(idx(i), dims(i)) = lb(dims(i)) + rand()*(ub(dims(i))-lb(dims(i)));
            end
        end
    end
end