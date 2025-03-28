function ans = getAnwser(pop,isCon,tmp,isFeedBack,problem,needinit,quesType,popLLM,num,modelName)    
%quesType---1:normal   2:feedBack   3:History
    global lastQues
    global lastAns
    global historyAnwser;
    global historyQues;
    global historyPop;
    ans = [];
    % 设置 API URL 和密钥
    apiKey = 'sk-VJRZh5bk7ZSc9PMRyUiGgZT3J4RcYUwQH0tegZgM8kCFiqzI';  % 替换为您的实际 API 密钥
    url = 'https://api.chatanywhere.tech/v1/chat/completions';
    % 设置请求头
    headers = {
        'Authorization', ['Bearer ' apiKey]; 
        'Content-Type', 'application/json'
    };
    decs = [];
    if ~isempty(pop)
        decs  = pop.decs;
    end
    D = problem.D;
    if isCon
        ques = getQuestionCon(pop); 
    else
        ques = getQuestionObj(pop,problem,num); 
    end
    if needinit
        ques = getQuestionInit(pop,problem,num);
    end
    if quesType == 1
        % 设置请求数据
        data = struct();
        data.model = modelName;  % 选择模型
        data.messages = {struct('role', 'user', 'content', ques)};
        data.temperature = tmp;  % 设置温度
    elseif quesType == 3
        if ~isempty(historyQues)
            messCell = {};
            for i = size(historyQues,1)
                messCell{end+1} = struct('role', 'user', 'content', historyQues{i});
                messCell{end+1} = struct('role', 'system', 'content', historyAnwser{i});
            end
            begin = 'Through my objective function calculation, I have obtained the true objective value of the solution you provided last time: \n ';
            back = 'I will describe my needs to you again:\n ';
            res = getSingle(popLLM);
            ques =  strcat(begin,res,back,ques);
            messCell{end+1} = struct('role', 'user', 'content', ques);
            
            % 设置请求数据
            data = struct();
            data.model = modelName;  % 选择模型
            data.messages = messCell;
            data.temperature = tmp;  % 设置温度
        else
            % 设置请求数据
            data = struct();
            data.model = modelName;  % 选择模型
            data.messages = {struct('role', 'user', 'content', ques)};
            data.temperature = tmp;  % 设置温度
        end
    else
        newQues = 'The solution you provided does not meet my requirements. Please note that generating identical solutions is prohibited. Prohibit generating solutions with the same value for each dimension. Prohibit generating solutions with all values being 0 or 1. Please regenerate 5 solutions. Each generated new solution must start with <start> and end with <end>. Do not write code. Do not provide any explanation.';
        % 设置请求数据
        data = struct();
        data.model = modelName;  % 选择模型
        data.messages = {struct('role', 'user', 'content', lastQues),struct('role', 'system', 'content', lastAns),struct('role', 'user', 'content', newQues)};
        data.temperature = tmp;  % 设置温度
    end
    % 将数据转为 JSON 格式
    jsonData = jsonencode(data);
    disp(jsonData);
    % 发送 POST 请求
    options = weboptions('HeaderFields', headers, 'MediaType', 'application/json', 'Timeout', 60);
    content = [];
    try
        response = webwrite(url, jsonData, options);
        %disp(response);
        % 提取 content
        content = response.choices(1).message.content;  % 使用大括号访问 cell 数组
        % 显示提取的内容
        content = string(content);
        disp(content);
        ans = exAns(content,D);
    catch e
        disp(['Error: ' e.message]);
    end
    if ~isFeedBack && ~isempty(content)
        lastQues = ques;
        lastAns = content;
    end
    if quesType == 3 && ~isempty(content)
        historyAnwser{end+1} = content;
        historyQues{end+1} = ques;
        historyPop{end+1}= popLLM;
    end
end

function outputArray =  exAns(str,D)
    % 使用正则表达式提取内容
    matches = regexp(str, '<start>(.*?)<end>', 'tokens');

    % 将提取的内容转换为二维数值数组
    numRows = length(matches);
    outputArray = zeros(numRows, D); % 假设每行有 D 个数值

    for i = 1:numRows
        % 将字符串转换为数值并存储
        outputArray(i, :) = str2num(matches{i}{1}); %#ok<ST2NM>
    end
end


function res = getSingle(pop)
      if isempty(pop)
            res = 'no  solution\n';
          return;
      end
      decs = pop.decs;
      objs = pop.objs;
      PopCon = pop.cons;
      CV = sum(max(0,PopCon),2);
      N = size(decs,1);
      res = '';
      for i = 1:N
          tmpD = decs(i,:);
          tmpP = objs(i,:);
          resD = join(string(tmpD), ',');
          
          resP = join(string(tmpP), ',');
          res = strcat(res,'   solution:','<start>',resD,'<end>\n');
          res = strcat(res,'   function values:',resP,'\n');
      end
end