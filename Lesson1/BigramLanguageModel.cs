namespace ukazure.llm.Lesson1;

internal sealed class BigramLanguageModel
{
    private readonly int[,] _counts;
    private readonly int _vocabSize;

    public BigramLanguageModel(int vocabSize)
    {
        _vocabSize = vocabSize;
        _counts = new int[vocabSize, vocabSize];
    }

    public void Train(List<(int Input, int Target)> pairs)
    {
        foreach (var pair in pairs)
        {
            _counts[pair.Input, pair.Target]++;
        }
    }

    public int PredictNext(int inputTokenId)
    {
        var bestTarget = 0;
        var bestCount = -1;

        for (var target = 0; target < _vocabSize; target++)
        {
            var count = _counts[inputTokenId, target];
            if (count > bestCount)
            {
                bestCount = count;
                bestTarget = target;
            }
        }

        return bestTarget;
    }

    public void PrintCounts(Dictionary<int, string> idToToken)
    {
        Console.WriteLine("Learned counts:");
        foreach (var input in Enumerable.Range(0, _vocabSize))
        {
            Console.Write($"{idToToken[input],-5} -> ");

            var parts = new List<string>();
            foreach (var target in Enumerable.Range(0, _vocabSize))
            {
                var count = _counts[input, target];
                if (count > 0)
                {
                    parts.Add($"{idToToken[target]} ({count})");
                }
            }

            Console.WriteLine(parts.Count > 0 ? string.Join(", ", parts) : "(none)");
        }
    }
}
