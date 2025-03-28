% MATLAB Code
function [offspring] = updateFunc1336(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    epsilon = 1e-12;
    
    % 1. Sort and partition population
    [~, sorted_idx] = sortrows([popfits, cons], [1 2]);
    pop_sorted = popdecs(sorted_idx, :);
    fits_sorted = popfits(sorted_idx);
    cons_sorted = cons(sorted_idx);
    
    elite_num = max(2, ceil(0.2*NP));
    middle_num = ceil(0.6*NP);
    elite = pop_sorted(1:elite_num, :);
    middle = pop_sorted(elite_num+1:elite_num+middle_num, :);
    inferior = pop_sorted(elite_num+middle_num+1:end, :);
    center = mean(pop_sorted(1:ceil(NP/2), :);
    
    % 2. Calculate adaptive parameters
    f_max = max(popfits);
    f_min = min(popfits);
    c_max = max(abs(cons)) + epsilon;
    
    F1 = 0.5 * (1 + (popfits - f_min) ./ (f_max - f_min + epsilon));
    F2 = 0.3 * (1 - abs(cons) ./ c_max);
    CR = 0.9 - 0.5 * (abs(cons) ./ c_max);
    
    % 3. Generate offspring
    for i = 1:NP
        % Select random indices
        candidates = setdiff(1:NP, i);
        idxs = candidates(randperm(length(candidates), 4));
        
        % Determine mutation strategy based on rank
        if ismember(i, sorted_idx(1:elite_num))
            % Elite mutation
            mutant = elite(randi(elite_num), :) + ...
                    F1(i) .* (pop_sorted(1,:) - popdecs(i,:)) + ...
                    F2(i) .* (popdecs(idxs(1),:) - popdecs(idxs(2),:));
        elseif ismember(i, sorted_idx(elite_num+1:elite_num+middle_num))
            % Middle mutation
            mutant = popdecs(i,:) + ...
                    F1(i) .* (popdecs(idxs(1),:) - popdecs(idxs(2),:)) + ...
                    F2(i) .* (popdecs(idxs(3),:) - popdecs(idxs(4),:));
        else
            % Inferior mutation
            mutant = center + ...
                    F1(i) .* (popdecs(idxs(1),:) - popdecs(idxs(2),:)) + ...
                    F2(i) .* (popdecs(idxs(3),:) - popdecs(idxs(4),:));
        end
        
        % Crossover
        j_rand = randi(D);
        mask = rand(1,D) < CR(i);
        mask(j_rand) = true;
        trial = popdecs(i,:);
        trial(mask) = mutant(mask);
        
        % Constraint repair
        if cons(i) > 0
            beta = 0.8 * min(1, abs(cons(i))/c_max;
            elite_repair = elite(randi(elite_num), :);
            trial = (1-beta)*trial + beta*elite_repair;
        end
        
        offspring(i,:) = trial;
    end
    
    % Boundary handling
    for j = 1:D
        below = offspring(:,j) < lb(j);
        above = offspring(:,j) > ub(j);
        offspring(below,j) = lb(j) + rand(sum(below),1) .* (ub(j)-lb(j));
        offspring(above,j) = lb(j) + rand(sum(above),1) .* (ub(j)-lb(j));
    end
end