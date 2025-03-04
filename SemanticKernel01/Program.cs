using dotenv.net;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using OpenAI.Chat;
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
        internal static int totalInputTokensForSession = 0;
        internal static int totalOutputTokensForSession = 0;
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
                StopSequences = new List<string> { "Bye", "Thank you" },
                MaxTokens = maxTokens,
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

                    // The last message in the chunk has the usage metadata.
                    // https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream_options
                    if (x.Metadata?["Usage"] != null)
                    {
                        ChatTokenUsage usage = (ChatTokenUsage)x.Metadata["Usage"];

                        totalInputTokensForSession += usage.InputTokenCount;
                        totalOutputTokensForSession += usage.OutputTokenCount;

                        Console.WriteLine();
                        Console.WriteLine("🔹 Token Usage Details 🔹");
                        Console.WriteLine($"📥 Input Tokens: {usage.InputTokenCount}");
                        Console.WriteLine($"📤 Output Tokens: {usage.OutputTokenCount}");
                        Console.WriteLine("----------------------------");
                        Console.WriteLine($"🟢 Cached Token Count: {usage.InputTokenDetails.CachedTokenCount}");
                        Console.WriteLine($"🎙️ Audio Token Count (Input): {usage.InputTokenDetails.AudioTokenCount}");
                        Console.WriteLine($"🎙️ Audio Token Count (Output): {usage.OutputTokenDetails.AudioTokenCount}");
                        Console.WriteLine($"🧠 Reasoning Token Count: {usage.OutputTokenDetails.ReasoningTokenCount}");
                        Console.WriteLine($"❌ Rejected Prediction Tokens: {usage.OutputTokenDetails.RejectedPredictionTokenCount}");
                        Console.WriteLine($"✅ Accepted Prediction Tokens: {usage.OutputTokenDetails.AcceptedPredictionTokenCount}");
                        Console.WriteLine("----------------------------");
                        Console.WriteLine();
                    }
                });

                history.AddAssistantMessage(responseText);
                Console.WriteLine();
            }
        }
    }

    internal class SementicTestPlugin
    {
        [KernelFunction("exit_chat")]
        [Description("Exit the chat when the user asks to exit, or says Bye or Thank you.")]
        public void ExitChat()
        {
            Console.WriteLine("Total Input Tokens: " + Program.totalInputTokensForSession);
            Console.WriteLine("Total Output Tokens: " + Program.totalOutputTokensForSession);
            Console.WriteLine("Exiting chat...");
            Environment.Exit(0);
        }
    }
}
