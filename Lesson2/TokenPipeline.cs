namespace ukazure.llm.Lesson2;

internal static class TokenPipeline
{
    public static List<string> Tokenize(string text)
    {
        return text
            .ToLowerInvariant()
            .Split([' ', '\r', '\n', '\t'], StringSplitOptions.RemoveEmptyEntries)
            .ToList();
    }

    public static List<string> BuildVocabulary(List<string> tokens)
    {
        return tokens
            .Distinct()
            .OrderBy(token => token)
            .ToList();
    }

    public static List<(int Input, int Target)> BuildTrainingPairs(List<int> encodedTokens)
    {
        var pairs = new List<(int Input, int Target)>();

        for (var index = 0; index < encodedTokens.Count - 1; index++)
        {
            pairs.Add((encodedTokens[index], encodedTokens[index + 1]));
        }

        return pairs;
    }
}
