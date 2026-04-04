namespace ukazure.llm.Lesson5;

internal sealed class TinyTransformerBlockModel
{
    private readonly int _contextSize;
    private readonly int _dModel;
    private readonly int _dHidden;
    private readonly double[,] _tokenEmbeddings;
    private readonly double[,] _positionEmbeddings;
    private readonly double[,] _queryWeights;
    private readonly double[,] _keyWeights;
    private readonly double[,] _valueWeights;
    private readonly double[,] _ff1Weights;
    private readonly double[] _ff1Bias;
    private readonly double[,] _ff2Weights;
    private readonly double[] _ff2Bias;
    private readonly double[,] _outputWeights;
    private readonly double[] _outputBias;
    private readonly Random _random;
    private readonly double _scale;

    public TinyTransformerBlockModel(int vocabSize, int contextSize, int dModel, int dHidden)
    {
        VocabSize = vocabSize;
        _contextSize = contextSize;
        _dModel = dModel;
        _dHidden = dHidden;
        _tokenEmbeddings = new double[vocabSize, dModel];
        _positionEmbeddings = new double[contextSize, dModel];
        _queryWeights = new double[dModel, dModel];
        _keyWeights = new double[dModel, dModel];
        _valueWeights = new double[dModel, dModel];
        _ff1Weights = new double[dModel, dHidden];
        _ff1Bias = new double[dHidden];
        _ff2Weights = new double[dHidden, dModel];
        _ff2Bias = new double[dModel];
        _outputWeights = new double[dModel, vocabSize];
        _outputBias = new double[vocabSize];
        _random = new Random(23);
        _scale = 1.0 / Math.Sqrt(dModel);

        Initialize(_tokenEmbeddings);
        Initialize(_positionEmbeddings);
        Initialize(_queryWeights);
        Initialize(_keyWeights);
        Initialize(_valueWeights);
        Initialize(_ff1Weights);
        Initialize(_ff2Weights);
        Initialize(_outputWeights);
    }

    public int VocabSize { get; }

    public int ContextSize => _contextSize;

    public int ModelWidth => _dModel;

    public int HiddenWidth => _dHidden;

    public TransformerTrainingSummary Train(List<TransformerTrainingExample> examples, int epochs, double learningRate)
    {
        var checkpoints = new List<(int Epoch, double Loss)>();
        var initialLoss = AverageLoss(examples);
        checkpoints.Add((0, initialLoss));

        for (var epoch = 1; epoch <= epochs; epoch++)
        {
            foreach (var example in examples)
            {
                TrainOnExample(example, learningRate);
                ClipParameters(3.0);
            }

            if (epoch % 100 == 0 || epoch == epochs)
            {
                checkpoints.Add((epoch, AverageLoss(examples)));
            }
        }

        return new TransformerTrainingSummary
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

    public TransformerInspection Inspect(int[] context)
    {
        var cache = Forward(context);
        return new TransformerInspection
        {
            AttentionWeights = cache.AttentionWeights,
            Probabilities = Softmax(cache.Logits)
        };
    }

    private void TrainOnExample(TransformerTrainingExample example, double learningRate)
    {
        var cache = Forward(example.Context);
        var probabilities = Softmax(cache.Logits);

        var dLogits = new double[VocabSize];
        Array.Copy(probabilities, dLogits, VocabSize);
        dLogits[example.Target] -= 1.0;

        var dH2 = new double[_dModel];
        for (var dim = 0; dim < _dModel; dim++)
        {
            for (var target = 0; target < VocabSize; target++)
            {
                dH2[dim] += _outputWeights[dim, target] * dLogits[target];
            }
        }

        for (var dim = 0; dim < _dModel; dim++)
        {
            for (var target = 0; target < VocabSize; target++)
            {
                _outputWeights[dim, target] -= learningRate * cache.H2[dim] * dLogits[target];
            }
        }

        for (var target = 0; target < VocabSize; target++)
        {
            _outputBias[target] -= learningRate * dLogits[target];
        }

        var dH1 = new double[_dModel];
        Array.Copy(dH2, dH1, _dModel);
        var dFf = dH2;

        var dA1 = new double[_dHidden];
        for (var hidden = 0; hidden < _dHidden; hidden++)
        {
            for (var dim = 0; dim < _dModel; dim++)
            {
                dA1[hidden] += _ff2Weights[hidden, dim] * dFf[dim];
            }
        }

        for (var hidden = 0; hidden < _dHidden; hidden++)
        {
            for (var dim = 0; dim < _dModel; dim++)
            {
                _ff2Weights[hidden, dim] -= learningRate * cache.A1[hidden] * dFf[dim];
            }
        }

        for (var dim = 0; dim < _dModel; dim++)
        {
            _ff2Bias[dim] -= learningRate * dFf[dim];
        }

        var dZ1 = new double[_dHidden];
        for (var hidden = 0; hidden < _dHidden; hidden++)
        {
            dZ1[hidden] = cache.Z1[hidden] > 0.0 ? dA1[hidden] : 0.0;
        }

        for (var dim = 0; dim < _dModel; dim++)
        {
            for (var hidden = 0; hidden < _dHidden; hidden++)
            {
                dH1[dim] += _ff1Weights[dim, hidden] * dZ1[hidden];
            }
        }

        for (var dim = 0; dim < _dModel; dim++)
        {
            for (var hidden = 0; hidden < _dHidden; hidden++)
            {
                _ff1Weights[dim, hidden] -= learningRate * cache.H1[dim] * dZ1[hidden];
            }
        }

        for (var hidden = 0; hidden < _dHidden; hidden++)
        {
            _ff1Bias[hidden] -= learningRate * dZ1[hidden];
        }

        var dX = new double[_contextSize][];
        for (var position = 0; position < _contextSize; position++)
        {
            dX[position] = new double[_dModel];
        }

        for (var dim = 0; dim < _dModel; dim++)
        {
            dX[^1][dim] += dH1[dim];
        }

        var dValues = new double[_contextSize][];
        var dWeights = new double[_contextSize];
        for (var position = 0; position < _contextSize; position++)
        {
            dValues[position] = new double[_dModel];
            for (var dim = 0; dim < _dModel; dim++)
            {
                dValues[position][dim] = cache.AttentionWeights[position] * dH1[dim];
                dWeights[position] += dH1[dim] * cache.Values[position][dim];
            }
        }

        var weightedSum = 0.0;
        for (var position = 0; position < _contextSize; position++)
        {
            weightedSum += dWeights[position] * cache.AttentionWeights[position];
        }

        var dScores = new double[_contextSize];
        for (var position = 0; position < _contextSize; position++)
        {
            dScores[position] = cache.AttentionWeights[position] * (dWeights[position] - weightedSum);
        }

        var dQuery = new double[_dModel];
        var dKeys = new double[_contextSize][];
        var keyBackprops = new double[_contextSize][];
        var valueBackprops = new double[_contextSize][];
        for (var position = 0; position < _contextSize; position++)
        {
            dKeys[position] = new double[_dModel];
            keyBackprops[position] = new double[_dModel];
            valueBackprops[position] = new double[_dModel];
            for (var dim = 0; dim < _dModel; dim++)
            {
                dQuery[dim] += dScores[position] * _scale * cache.Keys[position][dim];
                dKeys[position][dim] += dScores[position] * _scale * cache.Query[dim];
            }

            keyBackprops[position] = MultiplyMatrixVector(_keyWeights, dKeys[position]);
            valueBackprops[position] = MultiplyMatrixVector(_valueWeights, dValues[position]);
        }

        var queryBackprop = MultiplyMatrixVector(_queryWeights, dQuery);

        for (var position = 0; position < _contextSize; position++)
        {
            var input = cache.X[position];

            for (var inputDim = 0; inputDim < _dModel; inputDim++)
            {
                for (var outputDim = 0; outputDim < _dModel; outputDim++)
                {
                    _keyWeights[inputDim, outputDim] -= learningRate * input[inputDim] * dKeys[position][outputDim];
                    _valueWeights[inputDim, outputDim] -= learningRate * input[inputDim] * dValues[position][outputDim];
                }
            }

            for (var dim = 0; dim < _dModel; dim++)
            {
                dX[position][dim] += keyBackprops[position][dim] + valueBackprops[position][dim];
            }
        }

        var lastInput = cache.X[^1];
        for (var inputDim = 0; inputDim < _dModel; inputDim++)
        {
            for (var outputDim = 0; outputDim < _dModel; outputDim++)
            {
                _queryWeights[inputDim, outputDim] -= learningRate * lastInput[inputDim] * dQuery[outputDim];
            }
        }

        for (var dim = 0; dim < _dModel; dim++)
        {
            dX[^1][dim] += queryBackprop[dim];
        }

        for (var position = 0; position < _contextSize; position++)
        {
            var tokenId = example.Context[position];
            for (var dim = 0; dim < _dModel; dim++)
            {
                _tokenEmbeddings[tokenId, dim] -= learningRate * dX[position][dim];
                _positionEmbeddings[position, dim] -= learningRate * dX[position][dim];
            }
        }
    }

    private TransformerCache Forward(int[] context)
    {
        ValidateContext(context);

        var x = new double[_contextSize][];
        for (var position = 0; position < _contextSize; position++)
        {
            x[position] = new double[_dModel];
            for (var dim = 0; dim < _dModel; dim++)
            {
                x[position][dim] = _tokenEmbeddings[context[position], dim] + _positionEmbeddings[position, dim];
            }
        }

        var query = MultiplyVectorMatrix(x[^1], _queryWeights);
        var keys = x.Select(row => MultiplyVectorMatrix(row, _keyWeights)).ToArray();
        var values = x.Select(row => MultiplyVectorMatrix(row, _valueWeights)).ToArray();

        var scores = new double[_contextSize];
        for (var position = 0; position < _contextSize; position++)
        {
            scores[position] = Dot(query, keys[position]) * _scale;
        }

        var attentionWeights = Softmax(scores);
        var attentionOutput = new double[_dModel];
        for (var position = 0; position < _contextSize; position++)
        {
            for (var dim = 0; dim < _dModel; dim++)
            {
                attentionOutput[dim] += attentionWeights[position] * values[position][dim];
            }
        }

        var h1 = new double[_dModel];
        for (var dim = 0; dim < _dModel; dim++)
        {
            h1[dim] = x[^1][dim] + attentionOutput[dim];
        }

        var z1 = new double[_dHidden];
        for (var hidden = 0; hidden < _dHidden; hidden++)
        {
            var value = _ff1Bias[hidden];
            for (var dim = 0; dim < _dModel; dim++)
            {
                value += h1[dim] * _ff1Weights[dim, hidden];
            }

            z1[hidden] = value;
        }

        var a1 = z1.Select(ReLU).ToArray();
        var ff = new double[_dModel];
        for (var dim = 0; dim < _dModel; dim++)
        {
            var value = _ff2Bias[dim];
            for (var hidden = 0; hidden < _dHidden; hidden++)
            {
                value += a1[hidden] * _ff2Weights[hidden, dim];
            }

            ff[dim] = value;
        }

        var h2 = new double[_dModel];
        for (var dim = 0; dim < _dModel; dim++)
        {
            h2[dim] = h1[dim] + ff[dim];
        }

        var logits = new double[VocabSize];
        for (var target = 0; target < VocabSize; target++)
        {
            var logit = _outputBias[target];
            for (var dim = 0; dim < _dModel; dim++)
            {
                logit += h2[dim] * _outputWeights[dim, target];
            }

            logits[target] = logit;
        }

        return new TransformerCache
        {
            X = x,
            Query = query,
            Keys = keys,
            Values = values,
            AttentionWeights = attentionWeights,
            H1 = h1,
            Z1 = z1,
            A1 = a1,
            H2 = h2,
            Logits = logits
        };
    }

    private double AverageLoss(List<TransformerTrainingExample> examples)
    {
        var total = 0.0;

        foreach (var example in examples)
        {
            var probabilities = Inspect(example.Context).Probabilities;
            total += -Math.Log(Math.Max(probabilities[example.Target], 1e-9));
        }

        return total / examples.Count;
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

        if (sum == 0.0 || double.IsNaN(sum) || double.IsInfinity(sum))
        {
            var uniform = 1.0 / logits.Length;
            for (var index = 0; index < exps.Length; index++)
            {
                exps[index] = uniform;
            }

            return exps;
        }

        for (var index = 0; index < logits.Length; index++)
        {
            exps[index] /= sum;
        }

        return exps;
    }

    private static double ReLU(double value) => value > 0.0 ? value : 0.0;

    private void ValidateContext(int[] context)
    {
        if (context.Length != _contextSize)
        {
            throw new ArgumentException($"Expected context length {_contextSize}, but got {context.Length}.", nameof(context));
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

    private void ClipParameters(double limit)
    {
        Clip(_tokenEmbeddings, limit);
        Clip(_positionEmbeddings, limit);
        Clip(_queryWeights, limit);
        Clip(_keyWeights, limit);
        Clip(_valueWeights, limit);
        Clip(_ff1Weights, limit);
        Clip(_ff2Weights, limit);
        Clip(_outputWeights, limit);
        Clip(_ff1Bias, limit);
        Clip(_ff2Bias, limit);
        Clip(_outputBias, limit);
    }

    private static void Clip(double[,] matrix, double limit)
    {
        for (var row = 0; row < matrix.GetLength(0); row++)
        {
            for (var column = 0; column < matrix.GetLength(1); column++)
            {
                matrix[row, column] = Math.Clamp(matrix[row, column], -limit, limit);
            }
        }
    }

    private static void Clip(double[] vector, double limit)
    {
        for (var index = 0; index < vector.Length; index++)
        {
            vector[index] = Math.Clamp(vector[index], -limit, limit);
        }
    }

    private sealed class TransformerCache
    {
        public required double[][] X { get; init; }
        public required double[] Query { get; init; }
        public required double[][] Keys { get; init; }
        public required double[][] Values { get; init; }
        public required double[] AttentionWeights { get; init; }
        public required double[] H1 { get; init; }
        public required double[] Z1 { get; init; }
        public required double[] A1 { get; init; }
        public required double[] H2 { get; init; }
        public required double[] Logits { get; init; }
    }
}
