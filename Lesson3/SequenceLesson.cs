namespace ukazure.llm.Lesson3;

internal sealed class SequenceLesson
{
    public required string Corpus { get; init; }
    public required IReadOnlyList<string> SelectedSentences { get; init; }
    public required string StartSeed { get; init; }
    public required List<string> AvailableSeeds { get; init; }
    public required List<string> Tokens { get; init; }
    public required List<string> Vocabulary { get; init; }
    public required Dictionary<string, int> TokenToId { get; init; }
    public required Dictionary<int, string> IdToToken { get; init; }
    public required List<int> EncodedTokens { get; init; }
    public required List<SequenceTrainingExample> TrainingExamples { get; init; }
    public required SequenceModel Model { get; init; }
    public required SequenceTrainingSummary Summary { get; init; }

    public static SequenceLesson Create(IReadOnlyList<string> selectedSentences, string? startSeed = null)
    {
        if (selectedSentences.Count == 0)
        {
            throw new ArgumentException("At least one sentence must be selected.", nameof(selectedSentences));
        }

        var corpus = string.Join(Environment.NewLine, selectedSentences);

        var tokenLines = SequenceTokenPipeline.TokenizeLines(corpus);
        var tokens = tokenLines.SelectMany(line => line).ToList();
        var vocabulary = SequenceTokenPipeline.BuildVocabulary(tokens);

        var tokenToId = vocabulary
            .Select((token, index) => new { token, index })
            .ToDictionary(x => x.token, x => x.index);

        var idToToken = tokenToId.ToDictionary(x => x.Value, x => x.Key);
        var encodedTokens = tokens.Select(token => tokenToId[token]).ToList();
        var encodedLines = tokenLines
            .Select(line => line.Select(token => tokenToId[token]).ToList())
            .ToList();
        var trainingExamples = SequenceTokenPipeline.BuildTrainingExamples(encodedLines, contextSize: 3);
        var availableSeeds = trainingExamples
            .Select(example => string.Join(" ", example.Context.Select(id => idToToken[id])))
            .Distinct()
            .OrderBy(seed => seed)
            .ToList();

        var model = new SequenceModel(vocabulary.Count, contextSize: 3, embeddingSize: 4);
        var summary = model.Train(trainingExamples, epochs: 300, learningRate: 0.25);
        var chosenStartSeed = startSeed is not null && availableSeeds.Contains(startSeed, StringComparer.Ordinal)
            ? startSeed
            : availableSeeds[0];

        return new SequenceLesson
        {
            Corpus = corpus,
            SelectedSentences = selectedSentences.ToList(),
            StartSeed = chosenStartSeed,
            AvailableSeeds = availableSeeds,
            Tokens = tokens,
            Vocabulary = vocabulary,
            TokenToId = tokenToId,
            IdToToken = idToToken,
            EncodedTokens = encodedTokens,
            TrainingExamples = trainingExamples,
            Model = model,
            Summary = summary
        };
    }
}
