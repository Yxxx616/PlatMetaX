model = openAIChat("You are a helpful assistant.",ModelName="deepseek-chat", APIKey = "sk-5282cc59c61242d0aaf7f0a0d4ffcc30");
messages = messageHistory;
messages = addUserMessage(messages,"What is the precise definition of a treble crochet stitch?");
[generatedText,completeOutput] = generate(model,messages);
messages = addResponseMessage(messages,completeOutput);
messages = addUserMessage(messages,"When was it first invented?");