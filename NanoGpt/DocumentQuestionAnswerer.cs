namespace ukazure.llm.NanoGpt;

internal static class DocumentQuestionAnswerer
{
    private static readonly HashSet<string> StopWords =
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "about",
        "for",
        "from",
        "did",
        "do",
        "does",
        "how",
        "have",
        "has",
        "in",
        "is",
        "it",
        "me",
        "not",
        "of",
        "on",
        "or",
        "please",
        "should",
        "that",
        "the",
        "tell",
        "to",
        "what",
        "when",
        "where",
        "why",
        "with",
        "without"
    ];

    public static DocumentAnswer Ask(string document, string question)
    {
        var questionTokens = Tokenize(question)
            .Where(token => !StopWords.Contains(token))
            .ToHashSet();

        if (questionTokens.Count == 0)
        {
            return new DocumentAnswer(
                "Ask with a few concrete words from the topic you want to inspect.",
                []);
        }

        var matches = document
            .Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(line => new
            {
                Text = line,
                Score = Tokenize(line).Intersect(questionTokens).Count()
            })
            .Where(match => match.Score > 0)
            .OrderByDescending(match => match.Score)
            .ThenBy(match => match.Text.Length)
            .Take(3)
            .Select(match => (match.Text, match.Score))
            .ToList();

        if (matches.Count == 0)
        {
            return new DocumentAnswer(
                "I could not find a matching line in the training document. Try using words that appear in the document.",
                []);
        }

        var answer = string.Join(" ", matches.Select(match => match.Text));
        return new DocumentAnswer(answer, matches);
    }

    private static IEnumerable<string> Tokenize(string text)
    {
        return text
            .ToLowerInvariant()
            .Split([' ', '.', ',', '?', '!', ';', ':', '-', '\r', '\n', '\t'], StringSplitOptions.RemoveEmptyEntries)
            .Select(Normalize);
    }

    private static string Normalize(string token)
    {
        if (token.Length > 4 && token.EndsWith('s'))
        {
            return token[..^1];
        }

        return token;
    }
}
