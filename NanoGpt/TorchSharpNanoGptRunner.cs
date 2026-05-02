using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ukazure.llm.NanoGpt;

internal static class TorchSharpNanoGptRunner
{
    private const string TrainingDocumentPath = "data/nanogpt-training.txt";

    public static NanoGptRun Run()
    {
        manual_seed(31);

        var document = File.ReadAllText(TrainingDocumentPath);
        var tokenizer = CharacterTokenizer.FromText(document);
        var encoded = tokenizer.Encode(document);
        var config = new NanoGptConfig(
            VocabularySize: tokenizer.VocabularySize,
            BlockSize: 32,
            LayerCount: 1,
            HeadCount: 1,
            EmbeddingSize: 64,
            FeedForwardSize: 128);

        using var model = new TorchSharpCharacterModel(config);
        var optimiser = optim.AdamW(model.parameters(), lr: 0.01);
        var checkpoints = Train(model, optimiser, encoded, config, steps: 80, batchSize: 16);
        var prompt = "az ";
        var sample = Sample(model, tokenizer, prompt, maxNewCharacters: 120, temperature: 0.9, topK: 8);

        return new NanoGptRun
        {
            TrainingDocumentPath = TrainingDocumentPath,
            TrainingDocument = document,
            TrainingDocumentPreview = string.Join(Environment.NewLine, document.Split(Environment.NewLine).Take(8)),
            Tokenizer = tokenizer,
            Config = config,
            Checkpoints = checkpoints,
            Prompt = prompt,
            Sample = sample
        };
    }

    private static List<(int Step, double Loss)> Train(
        TorchSharpCharacterModel model,
        optim.Optimizer optimiser,
        int[] encoded,
        NanoGptConfig config,
        int steps,
        int batchSize)
    {
        var checkpoints = new List<(int Step, double Loss)>();
        var random = new Random(71);

        for (var step = 1; step <= steps; step++)
        {
            var batch = BuildBatch(encoded, config, batchSize, random);
            using var logits = model.call(batch.Inputs);
            using var loss = functional.cross_entropy(logits, batch.Targets);

            optimiser.zero_grad();
            loss.backward();
            optimiser.step();

            if (step == 1 || step % 20 == 0 || step == steps)
            {
                checkpoints.Add((step, loss.ToSingle()));
            }

            batch.Inputs.Dispose();
            batch.Targets.Dispose();
        }

        return checkpoints;
    }

    private static (Tensor Inputs, Tensor Targets) BuildBatch(
        int[] encoded,
        NanoGptConfig config,
        int batchSize,
        Random random)
    {
        var inputValues = new float[batchSize * config.BlockSize * config.VocabularySize];
        var targetValues = new long[batchSize];

        for (var batch = 0; batch < batchSize; batch++)
        {
            var start = random.Next(0, encoded.Length - config.BlockSize - 1);
            for (var offset = 0; offset < config.BlockSize; offset++)
            {
                var tokenId = encoded[start + offset];
                var index = (batch * config.BlockSize * config.VocabularySize)
                    + (offset * config.VocabularySize)
                    + tokenId;
                inputValues[index] = 1.0f;
            }

            targetValues[batch] = encoded[start + config.BlockSize];
        }

        var inputs = tensor(inputValues).reshape(batchSize, config.BlockSize * config.VocabularySize);
        var targets = tensor(targetValues, dtype: ScalarType.Int64);

        return (inputs, targets);
    }

    private static string Sample(
        TorchSharpCharacterModel model,
        CharacterTokenizer tokenizer,
        string prompt,
        int maxNewCharacters,
        double temperature,
        int topK)
    {
        var random = new Random(83);
        var tokens = tokenizer.Encode(prompt).ToList();

        for (var step = 0; step < maxNewCharacters; step++)
        {
            var context = tokens.TakeLast(model.Config.BlockSize).ToList();
            while (context.Count < model.Config.BlockSize)
            {
                context.Insert(0, 0);
            }

            using var input = BuildSingleInput(context, model.Config);
            using var logits = model.call(input).squeeze(0);
            var probabilities = ToTopKDistribution(logits, temperature, topK);
            tokens.Add(SampleFromDistribution(probabilities, random));
        }

        return tokenizer.Decode(tokens);
    }

    private static Tensor BuildSingleInput(IReadOnlyList<int> context, NanoGptConfig config)
    {
        var inputValues = new float[config.BlockSize * config.VocabularySize];

        for (var offset = 0; offset < config.BlockSize; offset++)
        {
            var index = (offset * config.VocabularySize) + context[offset];
            inputValues[index] = 1.0f;
        }

        return tensor(inputValues).reshape(1, config.BlockSize * config.VocabularySize);
    }

    private static List<(int TokenId, double Probability)> ToTopKDistribution(
        Tensor logits,
        double temperature,
        int topK)
    {
        var values = logits.data<float>().ToArray()
            .Select((value, index) => new { Index = index, Value = value / Math.Max(temperature, 0.05) })
            .OrderByDescending(entry => entry.Value)
            .Take(topK)
            .ToList();

        var max = values.Max(entry => entry.Value);
        var exps = values
            .Select(entry => new { entry.Index, Weight = Math.Exp(entry.Value - max) })
            .ToList();
        var total = exps.Sum(entry => entry.Weight);

        return exps
            .Select(entry => (TokenId: entry.Index, Probability: entry.Weight / total))
            .ToList();
    }

    private static int SampleFromDistribution(List<(int TokenId, double Probability)> probabilities, Random random)
    {
        var roll = random.NextDouble();
        var running = 0.0;

        foreach (var entry in probabilities)
        {
            running += entry.Probability;
            if (roll <= running)
            {
                return entry.TokenId;
            }
        }

        return probabilities[^1].TokenId;
    }

    private sealed class TorchSharpCharacterModel : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> _network;

        public TorchSharpCharacterModel(NanoGptConfig config)
            : base(nameof(TorchSharpCharacterModel))
        {
            Config = config;
            _network = Sequential(
                ("input", Linear(config.BlockSize * config.VocabularySize, config.FeedForwardSize)),
                ("relu1", ReLU()),
                ("hidden", Linear(config.FeedForwardSize, config.EmbeddingSize)),
                ("relu2", ReLU()),
                ("output", Linear(config.EmbeddingSize, config.VocabularySize)));

            RegisterComponents();
        }

        public NanoGptConfig Config { get; }

        public override Tensor forward(Tensor input)
        {
            return _network.call(input);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _network.Dispose();
            }

            base.Dispose(disposing);
        }
    }
}
