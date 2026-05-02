# nanoGPT C# Port Plan

This note maps Andrej Karpathy's nanoGPT repo onto the current `ukazure.llm` teaching codebase.

The goal is not to clone every Python/PyTorch detail immediately. The useful next step is to turn our existing lessons into a more complete C# GPT teaching implementation, keeping the same developer-first style while borrowing nanoGPT's architecture and workflow.

## What nanoGPT Adds

nanoGPT is built around a small number of important files:

- `model.py` defines the GPT model: token embeddings, positional embeddings, stacked transformer blocks, causal self-attention, MLP, layer norm, logits, loss, and generation.
- `train.py` defines the training loop: config, dataset loading, batching, evaluation, checkpointing, learning-rate decay, AdamW, gradient clipping, mixed precision, and optional distributed training.
- `sample.py` handles inference: load a checkpoint or GPT-2 variant, prompt the model, apply temperature and top-k, and generate tokens.
- `data/*/prepare.py` scripts turn raw text into token id streams.

Our code already teaches most individual ideas in miniature:

- Lesson 1: tokenisation, vocabulary, next-token pairs
- Lesson 2: embeddings, logits, softmax, loss, gradient descent
- Lesson 3: fixed context windows
- Lesson 4: attention
- Lesson 5: a tiny transformer-style block
- Lesson 6: training vs inference and sampling
- Lesson 7: prompt construction and retrieval

The next stage is to move from one teaching block to a small but recognisable GPT implementation.

## Recommended Direction

Keep the current lessons intact. Add a separate `nanogpt` command that introduces "nanoGPT in C#" as the bridge from toy model to real GPT training loop.

That command should not try to train a production model. It should show the architecture and workflow:

- multiple transformer blocks instead of one
- causal self-attention across every position
- layer normalisation
- multi-head attention
- GELU or ReLU MLP
- logits for every position
- cross-entropy over a whole sequence
- train/validation split
- checkpoint save/load
- generation from a trained checkpoint

## Proposed Folder Structure

```text
NanoGpt/
  NanoGptCourse.cs
  NanoGptRun.cs
  NanoGptPresenter.cs
  NanoGptConfig.cs
  CharacterTokenizer.cs
  TorchSharpNanoGptRunner.cs
```

The names are deliberately plain. This repo is a teaching project, so the type names should tell a .NET developer what each piece owns.

## Architecture Mapping

### nanoGPT `GPTConfig` -> `NanoGptConfig`

The C# config should capture the same core dimensions:

```csharp
internal sealed record NanoGptConfig(
    int VocabularySize,
    int BlockSize,
    int LayerCount,
    int HeadCount,
    int EmbeddingSize,
    int FeedForwardSize,
    double Dropout,
    bool UseBias);
```

For a teaching CPU run, start much smaller than GPT-2:

```text
VocabularySize:   character tokenizer size
BlockSize:        64
LayerCount:       2
HeadCount:        2
EmbeddingSize:    64
FeedForwardSize:  256
Dropout:          0.0
UseBias:          true
```

### nanoGPT `GPT` -> `TorchSharpNanoGptRunner`

nanoGPT's model shape is:

```text
token embeddings
+ position embeddings
-> dropout
-> N transformer blocks
-> final layer norm
-> language-model head
```

Our lesson 5 model already has token embeddings, positional embeddings, attention, feed-forward, residuals, and an output head. The C# port should generalise that from one final-position block to a full sequence model.

The important change is shape:

```text
Current lesson 5:
  context -> prediction for final position only

nanoGPT-style:
  batch x sequence -> logits for every sequence position
```

For teaching, we can avoid a general-purpose tensor library at first and implement a very small tensor helper that supports the shapes we need.

### nanoGPT `Block` -> `TransformerBlock`

nanoGPT's block is:

```text
x = x + attention(layerNorm1(x))
x = x + mlp(layerNorm2(x))
```

The C# equivalent should be explicit:

```csharp
var normalisedForAttention = _attentionNorm.Forward(x);
var attended = _attention.Forward(normalisedForAttention);
var h = x.Add(attended);

var normalisedForMlp = _mlpNorm.Forward(h);
var fedForward = _feedForward.Forward(normalisedForMlp);
return h.Add(fedForward);
```

This is a good lesson moment because residual connections stop being a diagram and become ordinary composition.

### nanoGPT `CausalSelfAttention` -> `CausalSelfAttention`

Lesson 4 has attention for one prediction position. nanoGPT computes attention for every position and prevents positions from looking into the future.

That means the C# version needs:

- query, key, and value projections
- split into heads
- scaled dot-product attention
- causal mask
- concatenate heads
- output projection

The causal mask is the key teaching point:

```text
position 0 can see: 0
position 1 can see: 0, 1
position 2 can see: 0, 1, 2
position 3 can see: 0, 1, 2, 3
```

This is the difference between a normal sequence model and an autoregressive GPT.

### nanoGPT `MLP` -> `FeedForward`

nanoGPT uses:

```text
linear -> GELU -> linear -> dropout
```

For this repo, use either GELU or ReLU. GELU is closer to GPT; ReLU is easier to explain. A good compromise is:

- implement ReLU first in `NanoGpt`
- mention that nanoGPT uses GELU
- optionally add GELU in a later polish pass

### nanoGPT `generate` -> `Sampler`

Lesson 6 already teaches greedy decoding, temperature, and top-k. That can become a reusable `Sampler`.

The C# version should support:

- prompt token ids
- max new tokens
- temperature
- top-k
- deterministic seed
- context cropping to `BlockSize`

The behaviour should mirror nanoGPT's generation loop:

```text
while more tokens needed:
  crop context to block size
  forward model
  take logits at final position
  apply temperature
  apply top-k
  sample next token
  append token
```

## Training Workflow

The training workflow should borrow nanoGPT's shape but stay CPU-friendly:

```text
prepare text
split train/validation
encode characters into ids
create random batches
forward pass
cross-entropy loss
backward pass
AdamW update
periodic evaluation
checkpoint save
sample from checkpoint
```

Do not try to support distributed training, mixed precision, PyTorch compile equivalents, or GPT-2 checkpoint loading at first. Those are useful production features, but they would drown the teaching value.

## Data Preparation

Start with a character-level tokenizer, like nanoGPT's Shakespeare quick start.

Why character-level first:

- no BPE dependency
- tiny vocabulary
- easy to explain
- easy to inspect
- works on CPU

Add a simple data file later:

```text
data/azure-cli-mini.txt
```

Example content:

```text
az deployment group create with bicep
az deployment group create with json
az webapp deploy with package
az webapp deploy with zip
dotnet build src api project
github actions deploy to appservice
```

Then `CharacterTokenizer` can build:

```csharp
Dictionary<char, int> CharToId
Dictionary<int, char> IdToChar
int[] EncodedText
```

## Tensor Strategy

There are two viable routes.

### Option A: Teaching Tensor

Create a tiny `Tensor` wrapper over `double[]`:

```csharp
internal sealed class Tensor
{
    public int[] Shape { get; }
    public double[] Values { get; }
}
```

Pros:

- very transparent
- no heavy dependency
- ideal for the talk and article

Cons:

- slow
- manual gradients become tedious
- batching and backprop get verbose quickly

### Option B: TorchSharp

Use TorchSharp, the .NET bindings for PyTorch.

Pros:

- much closer to nanoGPT
- automatic differentiation
- real tensors
- AdamW, cross-entropy, layer norm, matmul all become manageable

Cons:

- heavier dependency
- more setup friction
- less transparent for a beginner audience

Recommendation update: the repo now uses the TorchSharp route for the separate `nanogpt` command. The current implementation starts with a compact character-level training loop over a document, then leaves the full causal attention block stack as the next practical extension.

## Concrete Implementation Steps

1. Add `nanogpt` to the CLI.
2. Add a character tokenizer and tiny Azure CLI corpus.
3. Add `NanoGptConfig`.
4. Refactor lesson 5 ideas into reusable block types.
5. Implement causal self-attention across all positions.
6. Add layer normalisation.
7. Add stacked transformer blocks.
8. Add logits for every sequence position.
9. Add sequence cross-entropy.
10. Add a simple trainer with train/validation loss.
11. Add checkpoint save/load as JSON or binary.
12. Add a sampler that mirrors nanoGPT's `generate`.

## What To Defer

These nanoGPT features should wait:

- GPT-2 checkpoint loading
- BPE tokenisation with `tiktoken`
- distributed training
- mixed precision
- CUDA/MPS device management
- WandB logging
- model flops utilisation
- large OpenWebText-style datasets

They matter in nanoGPT because it is built to train medium-sized GPTs efficiently. This repo is currently teaching the internals, so clarity wins first.

## Suggested Command Narrative

The command should be called:

```text
nanogpt: C# with TorchSharp
```

The flow:

1. nanoGPT is the next rung, not a different ladder.
2. Replace one block with a stack of blocks.
3. Predict every position, not just the final position.
4. Add causal masking so the model cannot cheat.
5. Add layer norm to stabilise the block.
6. Train on batches, not single examples.
7. Save a checkpoint.
8. Generate from the checkpoint.
9. Explain what is still missing compared with nanoGPT.

That keeps the course honest: we are not pretending this toy C# version is nanoGPT-performance. We are making the architecture legible to .NET developers.
