namespace ukazure.llm.Lesson3;

internal static class SequenceTokenPipeline
{
    public static List<List<string>> TokenizeLines(string text)
    {
        return text
            .Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(Tokenize)
            .Where(tokens => tokens.Count > 0)
            .ToList();
    }

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

    public static List<SequenceTrainingExample> BuildTrainingExamples(List<int> encodedTokens, int contextSize)
    {
        var examples = new List<SequenceTrainingExample>();

        for (var index = 0; index <= encodedTokens.Count - contextSize - 1; index++)
        {
            var context = encodedTokens.Skip(index).Take(contextSize).ToArray();
            var target = encodedTokens[index + contextSize];
            examples.Add(new SequenceTrainingExample(context, target));
        }

        return examples;
    }

    public static List<SequenceTrainingExample> BuildTrainingExamples(IEnumerable<List<int>> encodedLines, int contextSize)
    {
        var examples = new List<SequenceTrainingExample>();

        foreach (var line in encodedLines)
        {
            examples.AddRange(BuildTrainingExamples(line, contextSize));
        }

        return examples;
    }
}
