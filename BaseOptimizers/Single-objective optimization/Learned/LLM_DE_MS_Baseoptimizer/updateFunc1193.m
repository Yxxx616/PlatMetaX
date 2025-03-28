% MATLAB Code
function [offspring] = updateFunc1193(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Calculate adaptive weights
    feasible_mask = cons <= 0;
    n_feas = sum(feasible_mask);
    min_fit = min(popfits);
    max_fit = max(popfits);
    range_fit = max_fit - min_fit + eps;
    
    % Feasibility weight
    if n_feas > 0
        feas_cons = abs(cons(feasible_mask));
        w_feas = (n_feas/NP) * (1 - mean(feas_cons)/(max(abs(cons))+eps));
    else
        w_feas = 0;
    end
    
    % Best solution weight
    w_best = exp(-(popfits - min_fit)./(std(popfits)+eps));
    w_best = w_best / sum(w_best);
    
    % 2. Select reference points
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    if n_feas > 0
        x_feas = mean(popdecs(feasible_mask, :), 1);
    else
        x_feas = mean(popdecs, 1);
    end
    
    % 3. Generate distinct random indices
    idx = 1:NP;
    R = zeros(NP, 7);
    for i = 1:7
        R(:,i) = randperm(NP)';
        while any(R(:,i) == idx')
            R(:,i) = randperm(NP)';
        end
    end
    
    % 4. Mutation components
    F1 = 0.6 + 0.2 * randn(NP, 1);
    F2 = 0.5 + 0.3 * randn(NP, 1);
    F3 = 0.4 + 0.4 * randn(NP, 1);
    
    v_feas = x_feas + F1 .* (popdecs(R(:,1),:) - popdecs(R(:,2),:));
    v_best = x_best + F2 .* (popdecs(R(:,3),:) - popdecs(R(:,4),:));
    v_rand = popdecs(R(:,5),:) + F3 .* (popdecs(R(:,6),:) - popdecs(R(:,7),:));
    
    % 5. Weighted combination
    w_feas_vec = w_feas * ones(1, D);
    w_best_vec = w_best * ones(1, D);
    mutants = w_feas_vec .* v_feas + w_best_vec .* v_best + ...
              (1 - w_feas_vec - w_best_vec) .* v_rand;
    
    % 6. Adaptive crossover
    CR = 0.5 + 0.4 * (popfits - min_fit) / range_fit;
    mask = rand(NP, D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    
    % Ensure final bounds
    offspring = min(max(offspring, lb), ub);
end