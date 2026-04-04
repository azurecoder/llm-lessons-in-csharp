namespace ukazure.llm.Lesson2;

internal sealed class NeuralBigramLesson
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
    public required NeuralBigramModel Model { get; init; }
    public required TrainingSummary Summary { get; init; }

    public static NeuralBigramLesson Create(IReadOnlyList<string> selectedSentences, string? startToken = null)
    {
        if (selectedSentences.Count == 0)
        {
            throw new ArgumentException("At least one sentence must be selected.", nameof(selectedSentences));
        }

        var corpus = string.Join(Environment.NewLine, selectedSentences);

        var tokens = TokenPipeline.Tokenize(corpus);
        var vocabulary = TokenPipeline.BuildVocabulary(tokens);

        var tokenToId = vocabulary
            .Select((token, index) => new { token, index })
            .ToDictionary(x => x.token, x => x.index);

        var idToToken = tokenToId.ToDictionary(x => x.Value, x => x.Key);
        var encodedTokens = tokens.Select(token => tokenToId[token]).ToList();
        var trainingPairs = TokenPipeline.BuildTrainingPairs(encodedTokens);

        var model = new NeuralBigramModel(vocabulary.Count, embeddingSize: 4);
        var summary = model.Train(trainingPairs, epochs: 250, learningRate: 0.35);
        var chosenStartToken = startToken is not null && tokenToId.ContainsKey(startToken)
            ? startToken
            : vocabulary[0];

        return new NeuralBigramLesson
        {
            Corpus = corpus,
            SelectedSentences = selectedSentences.ToList(),
            StartToken = chosenStartToken,
            Tokens = tokens,
            Vocabulary = vocabulary,
            TokenToId = tokenToId,
            IdToToken = idToToken,
            EncodedTokens = encodedTokens,
            TrainingPairs = trainingPairs,
            Model = model,
            Summary = summary
        };
    }
}
