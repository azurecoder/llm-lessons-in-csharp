namespace ukazure.llm.Lesson1;

internal sealed class TinyLlmLesson
{
    public required string Corpus { get; init; }
    public required IReadOnlyList<string> SelectedSentences { get; init; }
    public required string StartToken { get; init; }
    public required List<string> Tokens { get; init; }
    public required List<string> Vocabulary { get; init; }
    public required Dictionary<string, int> TokenToId { get; init; }
    public required Dictionary<int, string> IdToToken { get; init; }
    public required List<int> EncodedTokens { get; init; }
    public required List<(int Input, int Target)> TrainingPairs { get; init; }
    public required BigramLanguageModel Model { get; init; }

    public static TinyLlmLesson Create(IReadOnlyList<string> selectedSentences, string? startToken = null)
    {
        var normalizedSentences = selectedSentences
            .Where(sentence => !string.IsNullOrWhiteSpace(sentence))
            .Select(sentence => sentence.Trim())
            .ToList();

        var corpus = string.Join(Environment.NewLine, normalizedSentences);

        var tokens = TextPipeline.Tokenize(corpus);
        var vocabulary = TextPipeline.BuildVocabulary(tokens);

        var tokenToId = vocabulary
            .Select((token, index) => new { token, index })
            .ToDictionary(x => x.token, x => x.index);

        var idToToken = tokenToId.ToDictionary(x => x.Value, x => x.Key);
        var encodedTokens = tokens.Select(token => tokenToId[token]).ToList();
        var trainingPairs = TextPipeline.BuildTrainingPairs(encodedTokens);

        var model = new BigramLanguageModel(vocabulary.Count);
        model.Train(trainingPairs);

        return new TinyLlmLesson
        {
            Corpus = corpus,
            SelectedSentences = normalizedSentences,
            StartToken = startToken ?? vocabulary.First(),
            Tokens = tokens,
            Vocabulary = vocabulary,
            TokenToId = tokenToId,
            IdToToken = idToToken,
            EncodedTokens = encodedTokens,
            TrainingPairs = trainingPairs,
            Model = model
        };
    }
}
