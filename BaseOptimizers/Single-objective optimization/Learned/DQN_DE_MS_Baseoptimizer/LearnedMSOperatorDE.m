function Offspring = LearnedMSOperatorDE(Problem,Parent1,Parent2,Parent3,Gbest,MSnum,Parameter)

    %% Parameter setting
    if nargin > 4
        [CR,F,proM,disM] = deal(Parameter{:});
    else
        [CR,F,proM,disM] = deal(1,0.5,1,20);
    end
    if isa(Parent1(1),'SOLUTION')
        evaluated = true;
        Parent1   = Parent1.decs;
        Parent2   = Parent2.decs;
        Parent3   = Parent3.decs;
    else
        evaluated = false;
    end
    [N,D] = size(Parent1);

    %% Differental evolution
    Site = rand(N,D) < CR;
    Offspring       = Parent1;
    % 根据MSnum选择不同的变异策略
    switch MSnum
        case 1 % DE/rand/1
            Offspring(Site) = Parent1(Site) + F * (Parent2(Site) - Parent3(Site));
        case 2 % DE/best/1
            flatGbest = repmat(Gbest.decs,N,1);
            Offspring(Site) = flatGbest(Site) + F * (Parent2(Site) - Parent3(Site));
        case 3 % DE/cur-to-best/1
            flatGbest = repmat(Gbest.decs,N,1);
            Offspring(Site) = Parent1(Site) + F * (flatGbest(Site) - Parent1(Site)) + F * (Parent2(Site) - Parent3(Site));
        case 4 % DE/rand-to-best/1
            flatGbest = repmat(Gbest.decs,N,1);
            Offspring(Site) = Parent1(Site) + F * (flatGbest(Site) - Parent1(Site)) + F * (Parent2(Site) - Parent3(Site));

        otherwise
            error('Invalid MSnum value. MSnum must be 1, 2, 3, or 4.');
    end

    %% Polynomial mutation
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    Site  = rand(N,D) < proM/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    if evaluated
        Offspring = Problem.Evaluation(Offspring);
    end
end