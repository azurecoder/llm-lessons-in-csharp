using System.Text.Json;
using OpenAI.Chat;

namespace ukazure.llm.Lesson7;

internal static class OpenAiGroundedResponder
{
    public const string PlaceholderApiKey = "paste-your-openai-api-key-here";
    private const string ConfigFileName = "lesson7.config.json";

    public static OpenAiResponseResult GenerateGroundedAnswer(RetrievalLesson lesson)
    {
        var apiKey = ResolveApiKey();

        if (apiKey == PlaceholderApiKey)
        {
            return new OpenAiResponseResult
            {
                UsedLiveModel = false,
                Explanation = $"No API key configured. Set OPENAI_API_KEY or update {ConfigFileName}.",
                Answer = lesson.GroundedAnswer
            };
        }

        try
        {
            var client = new ChatClient(model: lesson.SuggestedModel, apiKey: apiKey);
            ChatCompletion completion = client.CompleteChat(
                [
                    new SystemChatMessage("You are a helpful Azure architecture assistant. Answer using only the supplied retrieved context. Keep the answer concise and practical for a software developer."),
                    new UserChatMessage(lesson.PromptWithRetrieval)
                ]);

            var answer = string.Join(
                Environment.NewLine,
                completion.Content
                    .Select(part => part.Text)
                    .Where(text => !string.IsNullOrWhiteSpace(text)));

            if (string.IsNullOrWhiteSpace(answer))
            {
                answer = lesson.GroundedAnswer;
            }

            return new OpenAiResponseResult
            {
                UsedLiveModel = true,
                Explanation = $"Live answer generated with the OpenAI .NET SDK using model {lesson.SuggestedModel}.",
                Answer = answer
            };
        }
        catch (Exception ex)
        {
            return new OpenAiResponseResult
            {
                UsedLiveModel = false,
                Explanation = $"OpenAI SDK call failed: {ex.Message}",
                Answer = lesson.GroundedAnswer
            };
        }
    }

    private static string ResolveApiKey()
    {
        var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");

        if (!string.IsNullOrWhiteSpace(apiKey))
        {
            return apiKey;
        }

        var configPath = Path.Combine(Directory.GetCurrentDirectory(), ConfigFileName);
        if (!File.Exists(configPath))
        {
            return PlaceholderApiKey;
        }

        try
        {
            var json = File.ReadAllText(configPath);
            var config = JsonSerializer.Deserialize<Lesson7Config>(json);

            if (!string.IsNullOrWhiteSpace(config?.OpenAiApiKey))
            {
                return config.OpenAiApiKey;
            }
        }
        catch
        {
            return PlaceholderApiKey;
        }

        return PlaceholderApiKey;
    }

    private sealed class Lesson7Config
    {
        public string? OpenAiApiKey { get; init; }
    }
}
