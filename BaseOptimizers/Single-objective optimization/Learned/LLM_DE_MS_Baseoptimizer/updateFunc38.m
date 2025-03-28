function [offspring] = updateFunc38(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Sort population by fitness and get ranks
    [~, sorted_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(sorted_idx) = 1:NP;
    
    % Determine elite group (top 10%)
    elite_size = max(1, floor(0.1*NP));
    elite_idx = sorted_idx(1:elite_size);
    
    % Normalize constraint violations
    cons_norm = sqrt(sum(cons.^2, 2));
    cons_norm(cons_norm == 0) = 1; % avoid division by zero
    cons_dir = bsxfun(@rdivide, cons, cons_norm);
    
    for i = 1:NP
        % Select elite base vector
        x_elite = popdecs(elite_idx(randi(elite_size)), :);
        
        % Select two distinct random vectors
        candidates = randperm(NP, 2);
        while any(candidates == i)
            candidates = randperm(NP, 2);
        end
        x_r1 = popdecs(candidates(1), :);
        x_r2 = popdecs(candidates(2), :);
        
        % Calculate adaptive parameters
        F = 0.5 + 0.3 * (ranks(i)/NP);
        C = 0.5 * (1 - ranks(i)/NP);
        
        % Calculate difference vector norm
        diff_norm = norm(x_r1 - x_r2);
        
        % Generate mutant vector
        offspring(i,:) = x_elite + F*(x_r1 - x_r2) + ...
                        C*cons_dir(i,:)*diff_norm;
        
        % Apply opposition-based learning with 20% probability
        if rand() < 0.2
            lb = min(popdecs);
            ub = max(popdecs);
            offspring(i,:) = lb + ub - offspring(i,:);
        end
    end
end