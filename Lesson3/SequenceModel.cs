namespace ukazure.llm.Lesson3;

internal sealed class SequenceModel
{
    private readonly int _contextSize;
    private readonly int _embeddingSize;
    private readonly double[,] _embeddings;
    private readonly double[,] _linearWeights;
    private readonly double[] _linearBias;
    private readonly Random _random;

    public SequenceModel(int vocabSize, int contextSize, int embeddingSize)
    {
        VocabSize = vocabSize;
        _contextSize = contextSize;
        _embeddingSize = embeddingSize;
        _embeddings = new double[vocabSize, embeddingSize];
        _linearWeights = new double[contextSize * embeddingSize, vocabSize];
        _linearBias = new double[vocabSize];
        _random = new Random(11);

        Initialize(_embeddings);
        Initialize(_linearWeights);
    }

    public int VocabSize { get; }

    public int ContextSize => _contextSize;
    

    public int EmbeddingSize => _embeddingSize;

    public SequenceTrainingSummary Train(List<SequenceTrainingExample> examples, int epochs, double learningRate)
    {
        var checkpoints = new List<(int Epoch, double Loss)>();
        var initialLoss = AverageLoss(examples);

        checkpoints.Add((0, initialLoss));

        for (var epoch = 1; epoch <= epochs; epoch++)
        {
            foreach (var example in examples)
            {
                TrainOnExample(example, learningRate);
            }

            if (epoch % 60 == 0 || epoch == epochs)
            {
                checkpoints.Add((epoch, AverageLoss(examples)));
            }
        }

        return new SequenceTrainingSummary
        {
            InitialLoss = initialLoss,
            FinalLoss = AverageLoss(examples),
            Checkpoints = checkpoints
        };
    }

    public int PredictNext(int[] context)
    {
        var probabilities = PredictProbabilities(context);
        var bestToken = 0;
        var bestProbability = double.MinValue;

        for (var index = 0; index < probabilities.Length; index++)
        {
            if (probabilities[index] > bestProbability)
            {
                bestProbability = probabilities[index];
                bestToken = index;
            }
        }

        return bestToken;
    }

    public double[] PredictProbabilities(int[] context)
    {
        ValidateContext(context);
        var combined = FlattenContextEmbeddings(context);
        var logits = ComputeLogits(combined);
        return Softmax(logits);
    }

    private void TrainOnExample(SequenceTrainingExample example, double learningRate)
    {
        var combined = FlattenContextEmbeddings(example.Context);
        var logits = ComputeLogits(combined);
        var probabilities = Softmax(logits);

        var dLogits = new double[VocabSize];
        Array.Copy(probabilities, dLogits, VocabSize);
        dLogits[example.Target] -= 1.0;

        var dCombined = new double[combined.Length];
        for (var feature = 0; feature < combined.Length; feature++)
        {
            for (var target = 0; target < VocabSize; target++)
            {
                dCombined[feature] += _linearWeights[feature, target] * dLogits[target];
            }
        }

        for (var feature = 0; feature < combined.Length; feature++)
        {
            for (var target = 0; target < VocabSize; target++)
            {
                var gradient = combined[feature] * dLogits[target];
                _linearWeights[feature, target] -= learningRate * gradient;
            }
        }

        for (var target = 0; target < VocabSize; target++)
        {
            _linearBias[target] -= learningRate * dLogits[target];
        }

        for (var position = 0; position < _contextSize; position++)
        {
            var tokenId = example.Context[position];
            for (var dim = 0; dim < _embeddingSize; dim++)
            {
                var featureIndex = position * _embeddingSize + dim;
                _embeddings[tokenId, dim] -= learningRate * dCombined[featureIndex];
            }
        }
    }

    private double AverageLoss(List<SequenceTrainingExample> examples)
    {
        var total = 0.0;

        foreach (var example in examples)
        {
            var probabilities = PredictProbabilities(example.Context);
            total += -Math.Log(Math.Max(probabilities[example.Target], 1e-9));
        }

        return total / examples.Count;
    }

    private double[] FlattenContextEmbeddings(int[] context)
    {
        var combined = new double[_contextSize * _embeddingSize];

        for (var position = 0; position < _contextSize; position++)
        {
            for (var dim = 0; dim < _embeddingSize; dim++)
            {
                var featureIndex = position * _embeddingSize + dim;
                combined[featureIndex] = _embeddings[context[position], dim];
            }
        }

        return combined;
    }

    private double[] ComputeLogits(double[] combined)
    {
        var logits = new double[VocabSize];

        for (var target = 0; target < VocabSize; target++)
        {
            var logit = _linearBias[target];
            for (var feature = 0; feature < combined.Length; feature++)
            {
                logit += combined[feature] * _linearWeights[feature, target];
            }

            logits[target] = logit;
        }

        return logits;
    }

    private void ValidateContext(int[] context)
    {
        if (context.Length != _contextSize)
        {
            throw new ArgumentException($"Expected context length {_contextSize}, but got {context.Length}.", nameof(context));
        }
    }

    private static double[] Softmax(double[] logits)
    {
        var max = logits.Max();
        var exps = new double[logits.Length];
        var sum = 0.0;

        for (var index = 0; index < logits.Length; index++)
        {
            exps[index] = Math.Exp(logits[index] - max);
            sum += exps[index];
        }

        for (var index = 0; index < exps.Length; index++)
        {
            exps[index] /= sum;
        }

        return exps;
    }

    private void Initialize(double[,] matrix)
    {
        for (var row = 0; row < matrix.GetLength(0); row++)
        {
            for (var column = 0; column < matrix.GetLength(1); column++)
            {
                matrix[row, column] = (_random.NextDouble() - 0.5) * 0.2;
            }
        }
    }
}
