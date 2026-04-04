using ukazure.llm.Lesson5;

namespace ukazure.llm.Lesson6;

internal static class InferenceTextGenerator
{
    public static string ContinueGreedy(
        TinyTransformerBlockModel model,
        Dictionary<string, int> tokenToId,
        Dictionary<int, string> idToToken,
        string seedText,
        int maxNewTokens)
    {
        var result = ParseSeed(seedText, model.ContextSize);

        for (var step = 0; step < maxNewTokens; step++)
        {
            var context = result.TakeLast(model.ContextSize).Select(token => tokenToId[token]).ToArray();
            var next = model.PredictNext(context);
            result.Add(idToToken[next]);
        }

        return string.Join(" ", result);
    }

    public static string ContinueSampled(
        TinyTransformerBlockModel model,
        Dictionary<string, int> tokenToId,
        Dictionary<int, string> idToToken,
        string seedText,
        int maxNewTokens,
        double temperature,
        int topK,
        int randomSeed)
    {
        var result = ParseSeed(seedText, model.ContextSize);
        var random = new Random(randomSeed);

        for (var step = 0; step < maxNewTokens; step++)
        {
            var context = result.TakeLast(model.ContextSize).Select(token => tokenToId[token]).ToArray();
            var probabilities = model.Inspect(context).Probabilities;
            var next = Sample(probabilities, temperature, topK, random);
            result.Add(idToToken[next]);
        }

        return string.Join(" ", result);
    }

    public static IEnumerable<(string Token, double Probability)> TopPredictions(
        TinyTransformerBlockModel model,
        Dictionary<string, int> tokenToId,
        Dictionary<int, string> idToToken,
        string seedText,
        int count)
    {
        var context = ParseSeed(seedText, model.ContextSize).Select(token => tokenToId[token]).ToArray();
        var probabilities = model.Inspect(context).Probabilities;

        return Enumerable.Range(0, probabilities.Length)
            .Select(index => (Token: idToToken[index], Probability: probabilities[index]))
            .OrderByDescending(entry => entry.Probability)
            .Take(count);
    }

    private static int Sample(double[] probabilities, double temperature, int topK, Random random)
    {
        var adjusted = probabilities
            .Select((probability, index) => new
            {
                Index = index,
                Score = Math.Pow(Math.Max(probability, 1e-9), 1.0 / Math.Max(temperature, 0.05))
            })
            .OrderByDescending(entry => entry.Score)
            .Take(topK)
            .ToList();

        var sum = adjusted.Sum(entry => entry.Score);
        var roll = random.NextDouble() * sum;
        var running = 0.0;

        foreach (var entry in adjusted)
        {
            running += entry.Score;
            if (roll <= running)
            {
                return entry.Index;
            }
        }

        return adjusted[^1].Index;
    }

    private static List<string> ParseSeed(string seedText, int expectedTokenCount)
    {
        var result = seedText
            .Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(token => token.ToLowerInvariant())
            .ToList();

        if (result.Count != expectedTokenCount)
        {
            throw new ArgumentException($"Seed text must contain exactly {expectedTokenCount} tokens.", nameof(seedText));
        }

        return result;
    }
}
