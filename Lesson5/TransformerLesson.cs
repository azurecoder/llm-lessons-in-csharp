namespace ukazure.llm.Lesson5;

internal sealed class TransformerLesson
{
    public required string Corpus { get; init; }
    public required IReadOnlyList<string> SelectedSentences { get; init; }
    public required string StartSeed { get; init; }
    public required List<string> AvailableSeeds { get; init; }
    public required List<string> Tokens { get; init; }
    public required List<string> Vocabulary { get; init; }
    public required Dictionary<string, int> TokenToId { get; init; }
    public required Dictionary<int, string> IdToToken { get; init; }
    public required List<TransformerTrainingExample> TrainingExamples { get; init; }
    public required TinyTransformerBlockModel Model { get; init; }
    public required TransformerTrainingSummary Summary { get; init; }

    public static TransformerLesson Create(IReadOnlyList<string> selectedSentences, string? startSeed = null)
    {
        if (selectedSentences.Count == 0)
        {
            throw new ArgumentException("At least one sentence must be selected.", nameof(selectedSentences));
        }

        var corpus = string.Join(Environment.NewLine, selectedSentences);

        var tokenLines = TransformerTokenPipeline.TokenizeLines(corpus);
        var tokens = tokenLines.SelectMany(line => line).ToList();
        var vocabulary = TransformerTokenPipeline.BuildVocabulary(tokens);

        var tokenToId = vocabulary
            .Select((token, index) => new { token, index })
            .ToDictionary(x => x.token, x => x.index);

        var idToToken = tokenToId.ToDictionary(x => x.Value, x => x.Key);
        var encodedLines = tokenLines
            .Select(line => line.Select(token => tokenToId[token]).ToList())
            .ToList();
        var trainingExamples = TransformerTokenPipeline.BuildTrainingExamples(encodedLines, contextSize: 4);
        var availableSeeds = trainingExamples
            .Select(example => string.Join(" ", example.Context.Select(id => idToToken[id])))
            .Distinct()
            .OrderBy(seed => seed)
            .ToList();

        var model = new TinyTransformerBlockModel(vocabulary.Count, contextSize: 4, dModel: 8, dHidden: 12);
        var summary = model.Train(trainingExamples, epochs: 400, learningRate: 0.05);
        var chosenStartSeed = startSeed is not null && availableSeeds.Contains(startSeed, StringComparer.Ordinal)
            ? startSeed
            : availableSeeds[0];

        return new TransformerLesson
        {
            Corpus = corpus,
            SelectedSentences = selectedSentences.ToList(),
            StartSeed = chosenStartSeed,
            AvailableSeeds = availableSeeds,
            Tokens = tokens,
            Vocabulary = vocabulary,
            TokenToId = tokenToId,
            IdToToken = idToToken,
            TrainingExamples = trainingExamples,
            Model = model,
            Summary = summary
        };
    }
}
