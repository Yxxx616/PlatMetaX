% MATLAB Code
function [offspring] = updateFunc1574(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Constraint-aware fitness adjustment
    beta = 0.8;
    adj_fits = popfits + beta * max(0, cons);
    
    % 2. Population partitioning
    [~, sorted_idx] = sort(adj_fits);
    top_N = max(1, floor(0.3*NP));
    mid_N = floor(0.5*NP);
    top_idx = sorted_idx(1:top_N);
    mid_idx = sorted_idx(top_N+1:top_N+mid_N);
    bot_idx = sorted_idx(top_N+mid_N+1:end);
    x_best = popdecs(sorted_idx(1), :);
    
    % 3. Generate random indices (vectorized)
    rand_idx = zeros(NP, 4);
    for i = 1:NP
        avail = setdiff(1:NP, i);
        perm = avail(randperm(length(avail), 4));
        rand_idx(i,:) = perm;
    end
    r1 = rand_idx(:,1); r2 = rand_idx(:,2); r3 = rand_idx(:,3); r4 = rand_idx(:,4);
    
    % 4. Group-specific mutation
    offspring = zeros(NP, D);
    
    % Top group - exploitation with multiple differences
    F1 = 0.6; F2 = 0.2;
    offspring(top_idx,:) = repmat(x_best, length(top_idx), 1) + ...
        F1*(popdecs(r1(top_idx),:) - popdecs(r2(top_idx),:)) + ...
        F2*(popdecs(r3(top_idx),:) - popdecs(r4(top_idx),:));
    
    % Middle group - balanced exploration-exploitation
    F_mid = 0.8;
    offspring(mid_idx,:) = popdecs(mid_idx,:) + ...
        F_mid*(repmat(x_best, length(mid_idx), 1) - popdecs(mid_idx,:)) + ...
        F_mid*(popdecs(r1(mid_idx),:) - popdecs(r2(mid_idx),:));
    
    % Bottom group - exploration with Cauchy noise
    F_bot = 1.0;
    cauchy = tan(pi*(rand(length(bot_idx),D)-0.5);
    offspring(bot_idx,:) = popdecs(r1(bot_idx),:) + ...
        F_bot*(popdecs(r2(bot_idx),:) - popdecs(r3(bot_idx),:)) + ...
        0.2*cauchy.*(ub-lb);
    
    % 5. Adaptive crossover
    ranks = (1:NP)'/NP;
    CR = 0.3 + 0.6*(1 - ranks);
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    offspring = popdecs .* (~mask) + offspring .* mask;
    
    % 6. Boundary handling with reflection and perturbation
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    rnd_pert = 0.05*rand(NP,D).*(ub-lb);
    
    % Reflection with perturbation
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask) + rnd_pert(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask) - rnd_pert(ub_mask);
    
    % Final boundary enforcement
    offspring = min(max(offspring, lb), ub);
end