% MATLAB Code
function [offspring] = updateFunc1571(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Constraint-aware ranking
    alpha = 0.7;
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    norm_cons = (abs(cons) - min(abs(cons))) / (max(abs(cons)) - min(abs(cons)) + eps);
    ranks = alpha * norm_fits + (1-alpha) * (1 - norm_cons);
    [~, sorted_idx] = sort(ranks);
    
    % 2. Dynamic subpopulation division
    top_N = max(1, floor(0.2*NP));
    mid_N = floor(0.6*NP);
    top_idx = sorted_idx(1:top_N);
    mid_idx = sorted_idx(top_N+1:top_N+mid_N);
    bot_idx = sorted_idx(top_N+mid_N+1:end);
    
    % 3. Group-specific mutation
    offspring = zeros(NP, D);
    x_best = popdecs(sorted_idx(1), :);
    
    % Generate all required random indices first
    all_idx = 1:NP;
    r1 = zeros(NP,1); r2 = zeros(NP,1); r3 = zeros(NP,1); r4 = zeros(NP,1);
    for i = 1:NP
        avail = all_idx(all_idx ~= i);
        perm = avail(randperm(length(avail), 4));
        r1(i) = perm(1); r2(i) = perm(2); r3(i) = perm(3); r4(i) = perm(4);
    end
    
    % Top group mutation
    F1 = 0.5; F2 = 0.3;
    offspring(top_idx,:) = repmat(x_best, length(top_idx), 1) + ...
        F1*(popdecs(r1(top_idx),:) - popdecs(r2(top_idx),:)) + ...
        F2*(popdecs(r3(top_idx),:) - popdecs(r4(top_idx),:));
    
    % Middle group mutation
    F_mid = 0.8;
    offspring(mid_idx,:) = popdecs(mid_idx,:) + ...
        F_mid*(repmat(x_best, length(mid_idx), 1) - popdecs(mid_idx,:)) + ...
        F_mid*(popdecs(r1(mid_idx),:) - popdecs(r2(mid_idx),:));
    
    % Bottom group mutation
    F_bot = 0.8;
    offspring(bot_idx,:) = popdecs(r1(bot_idx),:) + ...
        F_bot*(popdecs(r2(bot_idx),:) - popdecs(r3(bot_idx),:));
    
    % 4. Adaptive crossover
    CR = 0.5 + 0.4*(1 - ranks);
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs .* (~mask) + offspring .* mask;
    
    % 5. Boundary handling
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = (lb(lb_mask) + popdecs(lb_mask)) / 2;
    offspring(ub_mask) = (ub(ub_mask) + popdecs(ub_mask)) / 2;
    
    % Final boundary enforcement
    offspring = min(max(offspring, lb), ub);
end