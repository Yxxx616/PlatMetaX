function appendToHistory(newEntry)
% 将新条目追加到 history.json 文件
% 输入参数 newEntry: 结构体，包含以下字段：
%   - iter    : 迭代次数
%   - metrics : 结构体（minV, FR, time）
%   - error   : 错误信息字符串

% 检查文件是否存在并读取历史记录
filePathandName = 'E:\yx\PlatMetaX\PlatMetaX\PlatMetaX\MetaOptimizers\EAmetaoptimizers\LLM_DE_MS_Metaoptimizer\history.json';
if exist(filePathandName, 'file')
    fid = fopen(filePathandName, 'r');
    raw = fread(fid, inf, 'char=>char');
    fclose(fid);
    
    try
        history = jsondecode(raw');
    catch
        % 文件内容无效时初始化为空数组
        history = [];
    end
else
    history = [];
end

% 确保 history 是结构体数组
if isempty(history)
    history = newEntry;
else
    % 检查历史记录结构是否与新条目一致
    if ~isequal(fieldnames(history(1)), fieldnames(newEntry))
        error('新条目结构不匹配历史记录格式');
    end
    history(end+1) = newEntry;
end

% 写入更新后的文件
jsonStr = jsonencode(history, 'PrettyPrint', true);
fid = fopen(filePathandName, 'w');
if fid == -1
    error('无法写入文件 history.json');
end
fprintf(fid, '%s', jsonStr);
fclose(fid);
end