# nanoGPT Technical Walkthrough

The `nanogpt` command is the practical C# track in this repo. It is inspired by Andrej Karpathy's nanoGPT project, but it is not a full transformer implementation yet.

The goal is to make the training workflow legible to .NET developers:

- read a training document
- build a character vocabulary
- encode text as token IDs
- build tensor batches
- run a model forward pass
- calculate cross-entropy loss
- run backpropagation
- update parameters with AdamW
- sample generated text
- ask grounded questions over the source document

## Training Data

The training document is [`data/nanogpt-training.txt`](../data/nanogpt-training.txt).

It is a condensed Azure developer guide text based on:

```text
https://docs.azure.cn/en-us/guides/developer/azure-developer-guide
```

The document gives the demo Azure and software-development vocabulary: App Service, Azure Functions, Azure Monitor, Azure CLI, SDKs, Resource Manager, Bicep, regions, containers, identity, and storage.

## Main Runner

[`TorchSharpNanoGptRunner`](../NanoGpt/TorchSharpNanoGptRunner.cs) owns the training workflow.

```csharp
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
```

`VocabularySize` is the number of unique characters in the document. `BlockSize` is the number of previous characters used as context. `EmbeddingSize` and `FeedForwardSize` control the size of the small neural network. The current `LayerCount` and `HeadCount` values are placeholders for the fuller transformer version.

## Character Tokenizer

[`CharacterTokenizer`](../NanoGpt/CharacterTokenizer.cs) maps characters to integer token IDs and back again.

```csharp
public static CharacterTokenizer FromText(string text)
{
    var vocabulary = text
        .Distinct()
        .OrderBy(character => character)
        .ToList();

    var charToId = vocabulary
        .Select((character, index) => new { character, index })
        .ToDictionary(entry => entry.character, entry => entry.index);

    var idToChar = charToId.ToDictionary(entry => entry.Value, entry => entry.Key);

    return new CharacterTokenizer(charToId, idToChar);
}
```

This is character-level tokenisation. It is easier to inspect than subword tokenisation and works well for a small teaching demo. A production GPT-style system would normally use a subword tokenizer.

## Batches

The model trains on context windows. Each example contains `BlockSize` characters as input and the next character as the target.

```csharp
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
```

The input uses one-hot encoding. The active character ID is represented by a `1.0f`; all other vocabulary positions are `0.0f`.

TorchSharp then turns the arrays into tensors:

```csharp
var inputs = tensor(inputValues).reshape(batchSize, config.BlockSize * config.VocabularySize);
var targets = tensor(targetValues, dtype: ScalarType.Int64);
```

A tensor is a multidimensional numerical array. In this code, batching lets the model train on several examples in one optimisation step.

## Model

The current model is a compact feed-forward network, not the final transformer stack.

```csharp
_network = Sequential(
    ("input", Linear(config.BlockSize * config.VocabularySize, config.FeedForwardSize)),
    ("relu1", ReLU()),
    ("hidden", Linear(config.FeedForwardSize, config.EmbeddingSize)),
    ("relu2", ReLU()),
    ("output", Linear(config.EmbeddingSize, config.VocabularySize)));

RegisterComponents();
```

`Linear` is a learned matrix multiplication plus bias. `ReLU` is a non-linear activation function. The final layer produces one raw score for each possible next character.

Those raw scores are called logits. Logits are not probabilities. They are unnormalised scores that later become probabilities during loss calculation or sampling.

## Training Loop

The core training loop is short because TorchSharp handles automatic differentiation.

```csharp
using var logits = model.call(batch.Inputs);
using var loss = functional.cross_entropy(logits, batch.Targets);

optimiser.zero_grad();
loss.backward();
optimiser.step();
```

`model.call` runs the forward pass. `cross_entropy` compares the predicted logits with the correct target token IDs. `zero_grad` clears previous gradients. `loss.backward` runs backpropagation. `optimiser.step` updates the model parameters.

AdamW is the optimiser:

```csharp
var optimiser = optim.AdamW(model.parameters(), lr: 0.01);
```

AdamW is an adaptive gradient-based optimiser with decoupled weight decay. The optimiser decides how to update each model parameter from the gradients calculated during backpropagation.

## Sampling

Generation repeatedly predicts one next token and appends it to the context.

```csharp
var context = tokens.TakeLast(model.Config.BlockSize).ToList();

while (context.Count < model.Config.BlockSize)
{
    context.Insert(0, 0);
}

using var input = BuildSingleInput(context, model.Config);
using var logits = model.call(input).squeeze(0);
var probabilities = ToTopKDistribution(logits, temperature, topK);
tokens.Add(SampleFromDistribution(probabilities, random));
```

`temperature` controls how conservative or varied the generated text is. Lower values make high-scoring tokens more dominant. Higher values flatten the distribution.

`topK` limits sampling to the best `k` candidates. The current command uses `topK: 8`, so the sampler only considers the eight highest-scoring next characters.

## Grounded Questions

The interactive question loop is separate from generation. The tiny character model is not used as a semantic question-answering system.

[`DocumentQuestionAnswerer`](../NanoGpt/DocumentQuestionAnswerer.cs) retrieves matching lines from the training document:

```csharp
var matches = document
    .Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
    .Select(line => new
    {
        Text = line,
        Score = Tokenize(line).Intersect(questionTokens).Count()
    })
    .Where(match => match.Score > 0)
    .OrderByDescending(match => match.Score)
    .ThenBy(match => match.Text.Length)
    .Take(3)
    .Select(match => (match.Text, match.Score))
    .ToList();
```

The score is a simple word-overlap score. This makes the retrieval step easy to explain, but it also shows why real retrieval-augmented generation systems usually use embeddings, vector search, better chunking, reranking, and a final LLM response step.

## Useful Demo Questions

These questions work well with the current Azure developer guide training text:

- What is App Service useful for?
- When should I use Azure Functions?
- How can developers manage Azure resources?
- What does Azure Monitor help with?
- Why would I use Bicep or ARM templates?

## What Comes Next

The next technical step is to replace the compact feed-forward model with a fuller nanoGPT-style transformer stack:

- token embeddings
- positional embeddings
- causal self-attention
- multi-head attention
- layer normalisation
- feed-forward blocks
- residual connections
- sequence-level cross-entropy
- checkpoint save/load

That would move the command closer to nanoGPT while keeping the repo focused on teaching C# developers how the pieces fit together.
