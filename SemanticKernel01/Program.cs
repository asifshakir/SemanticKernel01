using dotenv.net;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

namespace SemanticKernel01
{
    internal class Program
    {
        async static Task Main(string[] args)
        {
            var envVars = DotEnv.Read();

            var modelId = envVars["CHATGPT_MODEL"];
            var apiKey = envVars["CHATGPT_API_KEY"];
            var maxTokens = int.Parse(envVars["CHATGPT_MAX_TOKENS"]);

            IKernelBuilder kernelBuilder = Kernel.CreateBuilder();
            kernelBuilder
                .AddOpenAIChatCompletion(
                    modelId: modelId,
                    apiKey: apiKey
                );
            kernelBuilder.Plugins.AddFromType<SementicTestPlugin>();
            Kernel kernel = kernelBuilder.Build();

            var chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();

            var openAIPromptExecutionSettings = new OpenAIPromptExecutionSettings
            {
                FunctionChoiceBehavior = FunctionChoiceBehavior.Auto()
            };

            ChatHistory history = new ChatHistory();
            while (true)
            {
                Console.Write("You: ");
                var userMessage = Console.ReadLine();
                history.AddUserMessage(userMessage);
                var response = chatCompletionService.GetStreamingChatMessageContentsAsync(
                    chatHistory: history,
                    executionSettings: openAIPromptExecutionSettings,
                    kernel: kernel
                );

                var responseText = "";

                Console.WriteLine("The assistant is thinking...");
                Console.Write("Assistant: ");

                await response.ForEachAsync(x =>
                {
                    responseText += x.Content;
                    Console.Write(x.Content);
                });

                history.AddAssistantMessage(responseText);

                Console.WriteLine();
            }
        }
    }

    public class SementicTestPlugin
    {
        [KernelFunction("exit_chat")]
        [Description("Exit the chat, when the user ends the response with a Bye or Thank you.")]
        public void ExitChat()
        {
            Console.WriteLine("Exiting chat...");
            Environment.Exit(0);
        }
    }
}
