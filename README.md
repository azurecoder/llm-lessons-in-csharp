# ukazure.llm

A small C# console project that teaches how a large language model is built, one step at a time.

The repo is aimed at software developers rather than data scientists. Each lesson keeps the mechanics visible and uses tiny Azure and programming-flavoured examples so you can see what the code is doing without needing a GPU cluster or an existential crisis.

## What is in here

The interactive lessons currently cover:

- `lesson1` - a count-based bigram model
- `lesson2` - replacing counts with learnable weights
- `lesson3` - sequence windows and ordered context
- `lesson4` - attention
- `lesson5` - a tiny transformer-style block
- `lesson6` - training, inference, and sampling
- `lesson7` - prompting, retrieval, and grounded answers
- `nanogpt` - a TorchSharp-backed nanoGPT track using a training document

## Requirements

- .NET 10 SDK
- a terminal that supports interactive console input
- macOS users running `nanogpt` may need `brew install libomp` for TorchSharp's CPU backend

This project uses:

- [`Spectre.Console`](https://spectreconsole.net/) for the interactive lesson UI
- the official [`OpenAI`](https://www.nuget.org/packages/OpenAI) .NET SDK for the optional live model call in lesson 7
- [`TorchSharp-cpu`](https://www.nuget.org/packages/TorchSharp-cpu) for the practical nanoGPT command

## How to run

From the repo root:

```bash
dotnet run --project ukazure.llm.cli.csproj -- lesson1
```

Replace `lesson1` with any lesson name from `lesson1` to `nanogpt`.

If you run the app without a valid lesson argument, it prints the available lessons:

```bash
dotnet run --project ukazure.llm.cli.csproj
```

Example output:

```text
Usage: dotnet run lesson1|lesson2|lesson3|lesson4|lesson5|lesson6|lesson7|nanogpt

Available commands:
  lesson1  A tiny bigram model built from counts
  lesson2  Replace counts with learnable weights
  lesson3  Model sequencing with context windows
  lesson4  Attention lets the model choose what to focus on
  lesson5  A tiny transformer-style block
  lesson6  Training, inference, and sampling
  lesson7  Prompting, retrieval, and grounded answers
  nanogpt  nanoGPT in C# with TorchSharp
```

## Example lesson flow

### Lesson 1

Lesson 1 is interactive. You choose the training sentences, then pick a starting token for generation.

Example run:

```text
Choose the training sentences for lesson 1
  [x] azure deploys to the cloud
  [x] azure scales in the cloud
  [x] dotnet builds in the cloud
  [ ] dotnet runs in containers

Choose the starting token for generation
  azure
```

Representative output:

```text
Lesson 1: A tiny bigram model built from counts

Seed: azure
Current output: azure

Step 1: azure -> deploys
Current output: azure deploys

Step 2: deploys -> to
Current output: azure deploys to

Final output: azure deploys to the cloud
```

### Lesson 6

Lesson 6 uses the tiny transformer-style block from lesson 5 and shows the difference between training behaviour and inference behaviour.

Representative output:

```text
Lesson 6: Training, inference, and sampling

Top predictions after "az deployment group create":
  with         94.8%
  to            2.1%
  deploy        1.0%

Greedy output:
az deployment group create with

Temperature 0.7, top-k 3:
az deployment group create with bicep

Temperature 1.2, top-k 5:
az deployment group create with json
```

### Lesson 7

Lesson 7 moves from model internals to application architecture. You choose a developer question, the lesson retrieves relevant documents, builds a grounded prompt, and compares answers with and without retrieval.

Representative output:

```text
Choose the developer question for lesson 7
  How should I store secrets for my Azure app without hard-coding credentials?

Retrieved:
  Azure Key Vault
  Managed Identities
  Azure App Service

Without retrieval:
You should use a secure service for secrets, avoid hard-coded credentials, and prefer platform features that reduce direct secret handling.

With retrieval:
Store secrets in Azure Key Vault instead of appsettings files or source code. Use managed identities so the app can authenticate without storing passwords or client secrets. If the app runs on Azure App Service, configure app settings to reference Key Vault secrets.
```

### nanoGPT

The `nanogpt` command is separate from the numbered lessons. It is a C# rewrite track inspired by Andrej Karpathy's nanoGPT repo, using TorchSharp rather than hand-written arrays for the practical training machinery.

It reads a condensed training document from [`data/nanogpt-training.txt`](data/nanogpt-training.txt). The document is based on the Azure developer guide:

```text
https://docs.azure.cn/en-us/guides/developer/azure-developer-guide
```

TorchSharp provides tensors, automatic gradients, cross-entropy, and AdamW. The current implementation is intentionally small: it is a character-level model trained on Azure developer text, plus an interactive document-grounded question loop.

Representative output:

```text
nanoGPT: C# with TorchSharp

Path: data/nanogpt-training.txt

Vocabulary size: 60
Block size:     32
Embedding size: 64
Hidden size:    128

Step   1: loss = 4.1052
Step  20: loss = 3.1908
Step  40: loss = 3.0859
Step  60: loss = 2.8595
Step  80: loss = 2.5768

Prompt: az
Sample: az ...

Ask questions about data/nanogpt-training.txt.
Try: What is App Service useful for? or Why would I use Bicep or ARM templates?

Question: What is App Service useful for?
Answer: App Service is useful when a team wants a fast path to publish web projects. App Service for Linux can run custom container images for web applications. Hybrid Connections can connect an App Service application to on premises resources.
Evidence:
  App Service is useful when a team wants a fast path to publish web projects. (score 3)
  App Service for Linux can run custom container images for web applications. (score 2)
  Hybrid Connections can connect an App Service application to on premises resources. (score 2)
```

Submit a blank question to leave the question loop and finish the command. The generated sample still comes from the tiny TorchSharp character model; the question loop is deliberately grounded in the training document so the demo can answer useful questions without pretending that a small character model has suddenly become a semantic assistant.

Useful demo questions:

- What is App Service useful for?
- When should I use Azure Functions?
- How can developers manage Azure resources?
- What does Azure Monitor help with?
- Why would I use Bicep or ARM templates?

The evidence score is a simple lexical overlap score from the local document retriever. It is intentionally visible and imperfect, which makes it useful for explaining why production retrieval systems often add embeddings, chunking, reranking, and a final LLM answer-generation step.

For a deeper technical walkthrough of the current implementation, see [`docs/nanogpt-technical-walkthrough.md`](docs/nanogpt-technical-walkthrough.md).

## Lesson 7 configuration

Lesson 7 can optionally call a live model through the OpenAI .NET SDK.

By default, the lesson still works without any configuration. If no API key is present, it falls back to the locally composed grounded answer so the demo remains runnable.

The easiest way to enable the live call is to edit the local config file in the repo root:

```json
{
  "OpenAiApiKey": "your-api-key-here"
}
```

The file name is:

```text
lesson7.config.json
```

This file is ignored by git, so you can keep your local key there without committing it.

If you prefer, you can still use an environment variable instead:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Then run lesson 7:

```bash
dotnet run --project ukazure.llm.cli.csproj -- lesson7
```

If no key is configured, lesson 7 will show a message like:

```text
No API key configured. Set OPENAI_API_KEY or update lesson7.config.json.
```

## Notes

- The lessons are intentionally tiny and simplified.
- The goal is clarity, not scale or performance.
- Later lessons reuse ideas from earlier ones, so they work best as a sequence.
