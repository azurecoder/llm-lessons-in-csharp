namespace ukazure.llm.Lesson2;

internal static class NeuralTextGenerator
{
    public static string GenerateText(
        NeuralBigramModel model,
        Dictionary<string, int> tokenToId,
        Dictionary<int, string> idToToken,
        string startToken,
        int maxTokens)
    {
        var result = new List<string> { startToken };
        var current = tokenToId[startToken];

        for (var index = 0; index < maxTokens; index++)
        {
            var next = model.PredictNext(current);
            result.Add(idToToken[next]);
            current = next;
        }

        return string.Join(" ", result);
    }
}
