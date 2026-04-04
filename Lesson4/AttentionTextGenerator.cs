namespace ukazure.llm.Lesson4;

internal static class AttentionTextGenerator
{
    public static string ContinueText(
        AttentionModel model,
        Dictionary<string, int> tokenToId,
        Dictionary<int, string> idToToken,
        string seedText,
        int maxNewTokens)
    {
        var result = seedText
            .Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(token => token.ToLowerInvariant())
            .ToList();

        if (result.Count != model.ContextSize)
        {
            throw new ArgumentException($"Seed text must contain exactly {model.ContextSize} tokens.", nameof(seedText));
        }

        for (var step = 0; step < maxNewTokens; step++)
        {
            var context = result
                .TakeLast(model.ContextSize)
                .Select(token => tokenToId[token])
                .ToArray();

            var next = model.PredictNext(context);
            result.Add(idToToken[next]);
        }

        return string.Join(" ", result);
    }
}
