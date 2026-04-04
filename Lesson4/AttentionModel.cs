namespace ukazure.llm.Lesson4;

internal sealed class AttentionModel
{
    private readonly int _dModel;
    private readonly double[,] _embeddings;
    private readonly double[,] _queryWeights;
    private readonly double[,] _keyWeights;
    private readonly double[,] _valueWeights;
    private readonly double[,] _outputWeights;
    private readonly double[] _outputBias;
    private readonly Random _random;
    private readonly double _scale;

    public AttentionModel(int vocabSize, int contextSize, int dModel)
    {
        VocabSize = vocabSize;
        ContextSize = contextSize;
        _dModel = dModel;
        _embeddings = new double[vocabSize, dModel];
        _queryWeights = new double[dModel, dModel];
        _keyWeights = new double[dModel, dModel];
        _valueWeights = new double[dModel, dModel];
        _outputWeights = new double[dModel, vocabSize];
        _outputBias = new double[vocabSize];
        _random = new Random(17);
        _scale = 1.0 / Math.Sqrt(dModel);

        Initialize(_embeddings);
        Initialize(_queryWeights);
        Initialize(_keyWeights);
        Initialize(_valueWeights);
        Initialize(_outputWeights);
    }

    public int VocabSize { get; }

    public int ContextSize { get; }

    public int ModelWidth => _dModel;

    public AttentionTrainingSummary Train(List<AttentionTrainingExample> examples, int epochs, double learningRate)
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

            if (epoch % 80 == 0 || epoch == epochs)
            {
                checkpoints.Add((epoch, AverageLoss(examples)));
            }
        }

        return new AttentionTrainingSummary
        {
            InitialLoss = initialLoss,
            FinalLoss = AverageLoss(examples),
            Checkpoints = checkpoints
        };
    }

    public int PredictNext(int[] context)
    {
        var probabilities = Inspect(context).Probabilities;
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

    public AttentionInspection Inspect(int[] context)
    {
        ValidateContext(context);

        var x = GetInputEmbeddings(context);
        var q = MultiplyVectorMatrix(x[^1], _queryWeights);
        var keys = x.Select(row => MultiplyVectorMatrix(row, _keyWeights)).ToArray();
        var values = x.Select(row => MultiplyVectorMatrix(row, _valueWeights)).ToArray();

        var scores = new double[ContextSize];
        for (var position = 0; position < ContextSize; position++)
        {
            scores[position] = Dot(q, keys[position]) * _scale;
        }

        var weights = Softmax(scores);
        var contextVector = new double[_dModel];
        for (var position = 0; position < ContextSize; position++)
        {
            for (var dim = 0; dim < _dModel; dim++)
            {
                contextVector[dim] += weights[position] * values[position][dim];
            }
        }

        var logits = new double[VocabSize];
        for (var target = 0; target < VocabSize; target++)
        {
            var logit = _outputBias[target];
            for (var dim = 0; dim < _dModel; dim++)
            {
                logit += contextVector[dim] * _outputWeights[dim, target];
            }

            logits[target] = logit;
        }

        return new AttentionInspection
        {
            Weights = weights,
            Probabilities = Softmax(logits)
        };
    }

    private void TrainOnExample(AttentionTrainingExample example, double learningRate)
    {
        var x = GetInputEmbeddings(example.Context);
        var q = MultiplyVectorMatrix(x[^1], _queryWeights);
        var keys = x.Select(row => MultiplyVectorMatrix(row, _keyWeights)).ToArray();
        var values = x.Select(row => MultiplyVectorMatrix(row, _valueWeights)).ToArray();

        var scores = new double[ContextSize];
        for (var position = 0; position < ContextSize; position++)
        {
            scores[position] = Dot(q, keys[position]) * _scale;
        }

        var weights = Softmax(scores);
        var contextVector = new double[_dModel];
        for (var position = 0; position < ContextSize; position++)
        {
            for (var dim = 0; dim < _dModel; dim++)
            {
                contextVector[dim] += weights[position] * values[position][dim];
            }
        }

        var logits = new double[VocabSize];
        for (var target = 0; target < VocabSize; target++)
        {
            var logit = _outputBias[target];
            for (var dim = 0; dim < _dModel; dim++)
            {
                logit += contextVector[dim] * _outputWeights[dim, target];
            }

            logits[target] = logit;
        }

        var probabilities = Softmax(logits);
        var dLogits = new double[VocabSize];
        Array.Copy(probabilities, dLogits, VocabSize);
        dLogits[example.Target] -= 1.0;

        var dContextVector = new double[_dModel];
        for (var dim = 0; dim < _dModel; dim++)
        {
            for (var target = 0; target < VocabSize; target++)
            {
                dContextVector[dim] += _outputWeights[dim, target] * dLogits[target];
            }
        }

        for (var dim = 0; dim < _dModel; dim++)
        {
            for (var target = 0; target < VocabSize; target++)
            {
                _outputWeights[dim, target] -= learningRate * contextVector[dim] * dLogits[target];
            }
        }

        for (var target = 0; target < VocabSize; target++)
        {
            _outputBias[target] -= learningRate * dLogits[target];
        }

        var dValues = new double[ContextSize][];
        var dWeights = new double[ContextSize];
        for (var position = 0; position < ContextSize; position++)
        {
            dValues[position] = new double[_dModel];
            for (var dim = 0; dim < _dModel; dim++)
            {
                dValues[position][dim] = weights[position] * dContextVector[dim];
                dWeights[position] += dContextVector[dim] * values[position][dim];
            }
        }

        var weightedSum = 0.0;
        for (var position = 0; position < ContextSize; position++)
        {
            weightedSum += dWeights[position] * weights[position];
        }

        var dScores = new double[ContextSize];
        for (var position = 0; position < ContextSize; position++)
        {
            dScores[position] = weights[position] * (dWeights[position] - weightedSum);
        }

        var dQuery = new double[_dModel];
        var dKeys = new double[ContextSize][];

        for (var position = 0; position < ContextSize; position++)
        {
            dKeys[position] = new double[_dModel];
            for (var dim = 0; dim < _dModel; dim++)
            {
                dQuery[dim] += dScores[position] * _scale * keys[position][dim];
                dKeys[position][dim] += dScores[position] * _scale * q[dim];
            }
        }

        var dEmbeddings = new double[ContextSize][];
        for (var position = 0; position < ContextSize; position++)
        {
            dEmbeddings[position] = new double[_dModel];
        }

        for (var position = 0; position < ContextSize; position++)
        {
            var input = x[position];

            for (var inputDim = 0; inputDim < _dModel; inputDim++)
            {
                for (var outputDim = 0; outputDim < _dModel; outputDim++)
                {
                    _keyWeights[inputDim, outputDim] -= learningRate * input[inputDim] * dKeys[position][outputDim];
                    _valueWeights[inputDim, outputDim] -= learningRate * input[inputDim] * dValues[position][outputDim];
                }
            }

            var keyBackprop = MultiplyMatrixVector(_keyWeights, dKeys[position]);
            var valueBackprop = MultiplyMatrixVector(_valueWeights, dValues[position]);
            for (var dim = 0; dim < _dModel; dim++)
            {
                dEmbeddings[position][dim] += keyBackprop[dim] + valueBackprop[dim];
            }
        }

        var lastInput = x[^1];
        for (var inputDim = 0; inputDim < _dModel; inputDim++)
        {
            for (var outputDim = 0; outputDim < _dModel; outputDim++)
            {
                _queryWeights[inputDim, outputDim] -= learningRate * lastInput[inputDim] * dQuery[outputDim];
            }
        }

        var queryBackprop = MultiplyMatrixVector(_queryWeights, dQuery);
        for (var dim = 0; dim < _dModel; dim++)
        {
            dEmbeddings[^1][dim] += queryBackprop[dim];
        }

        for (var position = 0; position < ContextSize; position++)
        {
            var tokenId = example.Context[position];
            for (var dim = 0; dim < _dModel; dim++)
            {
                _embeddings[tokenId, dim] -= learningRate * dEmbeddings[position][dim];
            }
        }
    }

    private double AverageLoss(List<AttentionTrainingExample> examples)
    {
        var total = 0.0;

        foreach (var example in examples)
        {
            var probabilities = Inspect(example.Context).Probabilities;
            total += -Math.Log(Math.Max(probabilities[example.Target], 1e-9));
        }

        return total / examples.Count;
    }

    private double[][] GetInputEmbeddings(int[] context)
    {
        var x = new double[context.Length][];

        for (var position = 0; position < context.Length; position++)
        {
            x[position] = new double[_dModel];
            for (var dim = 0; dim < _dModel; dim++)
            {
                x[position][dim] = _embeddings[context[position], dim];
            }
        }

        return x;
    }

    private static double[] MultiplyVectorMatrix(double[] vector, double[,] matrix)
    {
        var result = new double[matrix.GetLength(1)];

        for (var column = 0; column < matrix.GetLength(1); column++)
        {
            for (var row = 0; row < matrix.GetLength(0); row++)
            {
                result[column] += vector[row] * matrix[row, column];
            }
        }

        return result;
    }

    private static double[] MultiplyMatrixVector(double[,] matrix, double[] vector)
    {
        var result = new double[matrix.GetLength(0)];

        for (var row = 0; row < matrix.GetLength(0); row++)
        {
            for (var column = 0; column < matrix.GetLength(1); column++)
            {
                result[row] += matrix[row, column] * vector[column];
            }
        }

        return result;
    }

    private static double Dot(double[] left, double[] right)
    {
        var total = 0.0;

        for (var index = 0; index < left.Length; index++)
        {
            total += left[index] * right[index];
        }

        return total;
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

        for (var index = 0; index < logits.Length; index++)
        {
            exps[index] /= sum;
        }

        return exps;
    }

    private void ValidateContext(int[] context)
    {
        if (context.Length != ContextSize)
        {
            throw new ArgumentException($"Expected context length {ContextSize}, but got {context.Length}.", nameof(context));
        }
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
