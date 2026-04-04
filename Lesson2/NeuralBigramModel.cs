namespace ukazure.llm.Lesson2;

internal sealed class NeuralBigramModel
{
    private readonly int _embeddingSize;
    private readonly double[,] _embeddings;
    private readonly double[,] _linearWeights;
    private readonly double[] _linearBias;
    private readonly Random _random;

    public NeuralBigramModel(int vocabSize, int embeddingSize)
    {
        VocabSize = vocabSize;
        _embeddingSize = embeddingSize;
        _embeddings = new double[vocabSize, embeddingSize];
        _linearWeights = new double[embeddingSize, vocabSize];
        _linearBias = new double[vocabSize];
        _random = new Random(7);

        Initialize(_embeddings);
        Initialize(_linearWeights);
    }

    public int VocabSize { get; }

    public int EmbeddingSize => _embeddingSize;

    public TrainingSummary Train(List<(int Input, int Target)> pairs, int epochs, double learningRate)
    {
        var checkpoints = new List<(int Epoch, double Loss)>();
        var initialLoss = AverageLoss(pairs);

        checkpoints.Add((0, initialLoss));

        for (var epoch = 1; epoch <= epochs; epoch++)
        {
            foreach (var pair in pairs)
            {
                TrainOnPair(pair.Input, pair.Target, learningRate);
            }

            if (epoch % 50 == 0 || epoch == epochs)
            {
                checkpoints.Add((epoch, AverageLoss(pairs)));
            }
        }

        var finalLoss = AverageLoss(pairs);

        return new TrainingSummary
        {
            InitialLoss = initialLoss,
            FinalLoss = finalLoss, 
            Checkpoints = checkpoints
        };
    }

    public int PredictNext(int inputTokenId)
    {
        var probabilities = PredictProbabilities(inputTokenId);
        var bestTarget = 0;
        var bestProbability = double.MinValue;

        for (var target = 0; target < VocabSize; target++)
        {
            if (!(probabilities[target] > bestProbability)) continue;
            bestProbability = probabilities[target];
            bestTarget = target;
        }

        return bestTarget;
    }

    public double[] PredictProbabilities(int inputTokenId)
    {
        var embedding = GetEmbedding(inputTokenId);
        var logits = ComputeLogits(embedding);
        return Softmax(logits);
    }

    private void TrainOnPair(int inputTokenId, int targetTokenId, double learningRate)
    {
        var embedding = GetEmbedding(inputTokenId);
        var logits = ComputeLogits(embedding);
        var probabilities = Softmax(logits);

        var dLogits = new double[VocabSize];
        Array.Copy(probabilities, dLogits, VocabSize);
        dLogits[targetTokenId] -= 1.0;

        var dEmbedding = new double[_embeddingSize];

        for (var dim = 0; dim < _embeddingSize; dim++)
        {
            for (var target = 0; target < VocabSize; target++)
            {
                dEmbedding[dim] += _linearWeights[dim, target] * dLogits[target];
            }
        }

        for (var dim = 0; dim < _embeddingSize; dim++)
        {
            for (var target = 0; target < VocabSize; target++)
            {
                var gradient = embedding[dim] * dLogits[target];
                _linearWeights[dim, target] -= learningRate * gradient;
            }
        }

        for (var target = 0; target < VocabSize; target++)
        {
            _linearBias[target] -= learningRate * dLogits[target];
        }

        for (var dim = 0; dim < _embeddingSize; dim++)
        {
            _embeddings[inputTokenId, dim] -= learningRate * dEmbedding[dim];
        }
    }

    private double AverageLoss(List<(int Input, int Target)> pairs)
    {
        var total = (from pair in pairs let probabilities = PredictProbabilities(pair.Input) select -Math.Log(Math.Max(probabilities[pair.Target], 1e-9))).Sum();

        return total / pairs.Count;
    }

    private double[] GetEmbedding(int tokenId)
    {
        var embedding = new double[_embeddingSize];

        for (var dim = 0; dim < _embeddingSize; dim++)
        {
            embedding[dim] = _embeddings[tokenId, dim];
        }

        return embedding;
    }

    private double[] ComputeLogits(double[] embedding)
    {
        var logits = new double[VocabSize];

        for (var target = 0; target < VocabSize; target++)
        {
            var logit = _linearBias[target];
            for (var dim = 0; dim < _embeddingSize; dim++)
            {
                logit += embedding[dim] * _linearWeights[dim, target];
            }

            logits[target] = logit;
        }

        return logits;
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
