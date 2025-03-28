% MATLAB Code
function [offspring] = updateFunc1192(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Calculate adaptive weights
    mean_fit = mean(popfits);
    std_fit = std(popfits) + eps;
    mean_cons = mean(abs(cons));
    std_cons = std(abs(cons)) + eps;
    
    w_fit = 1 ./ (1 + exp(-(popfits - mean_fit)/std_fit));
    w_cons = 1 ./ (1 + exp(abs(cons)/std_cons));
    w = w_fit .* w_cons;
    w = w / sum(w);
    
    % 2. Select reference points
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    feasible_mask = cons <= 0;
    if any(feasible_mask)
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
    F = 0.5 + 0.3 * randn(NP, 3);
    v_best = x_best + F(:,1) .* (popdecs(R(:,1),:) - popdecs(R(:,2),:));
    v_feas = x_feas + F(:,2) .* (popdecs(R(:,3),:) - popdecs(R(:,4),:));
    v_rand = popdecs(R(:,5),:) + F(:,3) .* (popdecs(R(:,6),:) - popdecs(R(:,7),:));
    
    % 5. Hybrid combination
    alpha = sum(feasible_mask)/NP;
    v_combined = alpha * v_feas + (1-alpha) * v_rand;
    mutants = bsxfun(@times, w, v_best) + bsxfun(@times, (1-w), v_combined);
    
    % 6. Dynamic crossover
    CR = 0.9 * w + 0.1 * (1-w);
    mask = rand(NP, D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Boundary handling with bounce-back
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = lb(lb_mask) + rand(sum(lb_mask(:)),1) .* ...
                        (popdecs(lb_mask) - lb(lb_mask));
    offspring(ub_mask) = ub(ub_mask) - rand(sum(ub_mask(:)),1) .* ...
                        (ub(ub_mask) - popdecs(ub_mask));
end