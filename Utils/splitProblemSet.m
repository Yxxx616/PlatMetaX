function [trainingSet,testingSet] = splitProblemSet(problemset)
%SPLITPROBLEMSET 此处显示有关此函数的摘要
pSetting = problemset.pSetting;
switch problemset.psName
    case 'TSPs'
        tspInstancesDir = 'TSP_Instances';
        if ~exist(tspInstancesDir, 'dir')
            mkdir(tspInstancesDir);
        end
        instanceFiles = dir(fullfile(tspInstancesDir, 'TSP_Instance_*.mat'));
        if length(instanceFiles) <= 0
            numInstances = 100;
            cityRange = [3,100];
            for i = 1 : numInstances
                numCities = randi(cityRange);
                tspInstance = TSPs();
                tspInstance.generateIns(numCities);
                filename = fullfile(tspInstancesDir, sprintf('TSP_Instance_%d.mat', i));
                tspInstance.saveInstance(filename);
                if i <= 70
                    trainingSet{i} = tspInstance;
                else
                    testingSet{i-350} = tspInstance;
                end
            end
        else
            numInstances = 100; %length(instanceFiles);
            for i = 1:numInstances
                data = load(fullfile(tspInstancesDir, instanceFiles(i).name));
                tspInstance = data.obj;
                if i <= floor(numInstances*0.7)
                    trainingSet{i} = tspInstance;
                else
                    testingSet{i-floor(numInstances*0.7)} = tspInstance;
                end
            end
        end   
    case 'LIRCMOP'
        for i = 1:14
            pName = sprintf('LIRCMOP%d', i);
            if i <= 10
                trainingSet{i} = eval([pName '(pSetting{:})']);
            else
                testingSet{i-10} = eval([pName '(pSetting{:})']);
            end
        end
    case 'BBOB'
        testList = [1,5,6,10,15,20];
        count1 = 1;
        count2 = 1;
        for i = 1:24 
            pName = sprintf('BBOB_F%d', i);
            if ismember(i, testList)
                testingSet{count1} = eval([pName '(pSetting{:})']);
                count1 = count1 + 1;
            else
                trainingSet{count2} = eval([pName '(pSetting{:})']);
                count2 = count2 + 1;
            end
        end
    case 'BBOBEC'
        for i = 1:24 
            pName = sprintf('BBOB_F%d', i);
            trainingSet{i} = eval([pName '(pSetting{:})']);
        end
        testingSet = trainingSet;
    case 'CF'
        for i = 1:10
            pName = sprintf('CF%d', i);
            if i <= 8
                trainingSet{i} = eval([pName '(pSetting{:})']);
            else
                testingSet{i-8} = eval([pName '(pSetting{:})']);
            end
        end
    case 'MW'
        for i = 1:14
            pName = sprintf('MW%d', i);
            if i <= 10
                trainingSet{i} = eval([pName '(pSetting{:})']);
            else
                testingSet{i-10} = eval([pName '(pSetting{:})']);
            end
        end
    case 'CEC2017'
        for i = 1:28
            pName = sprintf('CEC2017_F%d', i);
            if i <= 21
                trainingSet{i} = eval([pName '(pSetting{:})']);
            else
                testingSet{i-21} = eval([pName '(pSetting{:})']);
            end
        end
    otherwise
        trainingSet{1} = problemset;
        testingSet{1} = problemset;
end
        
end

