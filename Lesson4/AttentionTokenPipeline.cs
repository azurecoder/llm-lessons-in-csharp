namespace ukazure.llm.Lesson4;

internal static class AttentionTokenPipeline
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

    public static List<AttentionTrainingExample> BuildTrainingExamples(IEnumerable<List<int>> encodedLines, int contextSize)
    {
        var examples = new List<AttentionTrainingExample>();

        foreach (var line in encodedLines)
        {
            for (var index = 0; index <= line.Count - contextSize - 1; index++)
            {
                var context = line.Skip(index).Take(contextSize).ToArray();
                var target = line[index + contextSize];
                examples.Add(new AttentionTrainingExample(context, target));
            }
        }

        return examples;
    }
}
