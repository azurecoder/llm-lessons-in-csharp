using Spectre.Console;

namespace ukazure.llm.Lesson6;

internal static class InferenceLessonPresenter
{
    public static void Run(InferenceLesson lesson)
    {
        Console.Clear();

        ShowStep(
            "Lesson 6: Training, inference, and sampling",
            [
                "We now have a tiny transformer-style block.",
                "The next question is operational: how does a model go from training to useful output?",
                "This lesson focuses on the runtime behaviour software developers actually care about."
            ]);

        Pause();

        ShowStep(
            "1. Same model, two modes",
            [
                "Training mode updates weights.",
                "Inference mode only runs the forward pass.",
                "That split is a lot like build time vs run time."
            ]);

        Pause();

        ShowStep(
            "2. A tiny command-style corpus",
            [
                "We train on Azure CLI and developer-flavoured text.",
                "You can choose the training sentences first, just like lesson 5.",
                "That keeps the outputs familiar to you."
            ]);
        AnsiConsole.MarkupLine($"[deepskyblue1]{Markup.Escape(lesson.Corpus)}[/]");

        Pause();

        ShowStep(
            "3. Training still means next-token prediction",
            [
                "Even at this stage, the API has not changed.",
                "We still feed a context window in and ask for the next token."
            ]);
        foreach (var example in lesson.TrainingExamples.Take(6))
        {
            var context = string.Join(" ", example.Context.Select(id => lesson.IdToToken[id]));
            AnsiConsole.MarkupLine($"[aqua][[{Markup.Escape(context)}]][/][grey] ->[/] [springgreen3]{Markup.Escape(lesson.IdToToken[example.Target])}[/]");
        }

        if (lesson.TrainingExamples.Count > 6)
        {
            AnsiConsole.MarkupLine($"[grey]... and {lesson.TrainingExamples.Count - 6} more training examples[/]");
        }

        Pause();

        ShowStep(
            "4. What happens during training?",
            [
                "Forward pass: compute probabilities for the next token.",
                "Loss: compare the prediction to the expected token.",
                "Backpropagation: compute gradients.",
                "Optimiser step: update weights."
            ]);
        AnsiConsole.MarkupLine($"[grey]Initial loss:[/] [aqua]{lesson.Summary.InitialLoss:F4}[/]");
        AnsiConsole.MarkupLine($"[grey]Final loss:[/]   [springgreen3]{lesson.Summary.FinalLoss:F4}[/]");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[grey]Checkpoints:[/]");
        foreach (var checkpoint in lesson.Summary.Checkpoints)
        {
            AnsiConsole.MarkupLine($"[grey]  Epoch {checkpoint.Epoch,3}:[/] [aqua]loss = {checkpoint.Loss:F4}[/]");
        }

        Pause();

        ShowStep(
            "5. What happens during inference?",
            [
                "Forward pass only.",
                "No gradients.",
                "No weight updates.",
                "Just predict one token, append it, and repeat."
            ]);
        AnsiConsole.MarkupLine("[grey]Software-engineering translation:[/]");
        AnsiConsole.MarkupLine("[aqua]- Training mutates model state[/]");
        AnsiConsole.MarkupLine("[aqua]- Inference treats the model as a read-only service[/]");

        Pause();

        ShowStep(
            "6. Greedy decoding",
            [
                "The simplest inference strategy is greedy decoding.",
                "At each step, take the highest-probability next token."
            ]);
        PrintTopPredictions(lesson, lesson.StartSeed, 5);
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[grey]Greedy output:[/]");
        AnsiConsole.MarkupLine(
            $"[springgreen3]{Markup.Escape(InferenceTextGenerator.ContinueGreedy(lesson.Model, lesson.TokenToId, lesson.IdToToken, lesson.StartSeed, 1))}[/]");

        Pause();

        ShowStep(
            "7. Sampling makes output less rigid",
            [
                "Greedy decoding is stable but repetitive.",
                "Sampling lets us choose from the probability distribution instead of always taking the top result."
            ]);
        AnsiConsole.MarkupLine("[grey]Temperature 0.7, top-k 3:[/]");
        AnsiConsole.MarkupLine(
            $"[springgreen3]{Markup.Escape(InferenceTextGenerator.ContinueSampled(lesson.Model, lesson.TokenToId, lesson.IdToToken, lesson.StartSeed, 1, temperature: 0.7, topK: 3, randomSeed: 7))}[/]");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[grey]Temperature 1.2, top-k 5:[/]");
        AnsiConsole.MarkupLine(
            $"[springgreen3]{Markup.Escape(InferenceTextGenerator.ContinueSampled(lesson.Model, lesson.TokenToId, lesson.IdToToken, lesson.StartSeed, 1, temperature: 1.2, topK: 5, randomSeed: 19))}[/]");

        Pause();

        ShowStep(
            "8. Temperature is a behaviour knob",
            [
                "Lower temperature sharpens the distribution.",
                "Higher temperature flattens it.",
                "Same model weights, different runtime behaviour."
            ]);
        AnsiConsole.MarkupLine("[grey]Developer translation:[/]");
        AnsiConsole.MarkupLine("[aqua]- temperature is a runtime config setting[/]");
        AnsiConsole.MarkupLine("[aqua]- the model binary is unchanged[/]");
        AnsiConsole.MarkupLine("[aqua]- only the token selection policy changes[/]");

        Pause();

        ShowStep(
            "9. Real models stack many blocks",
            [
                "This demo still uses one tiny transformer-style block in code.",
                "A production LLM stacks many blocks of the same shape.",
                "That is more like a deep service pipeline than a single function call."
            ]);

        Pause();

        ShowStep(
            "10. The key mental model",
            [
                "Pretraining teaches the model broad statistical patterns.",
                "Inference turns those learned weights into token-by-token behaviour.",
                "Sampling controls how deterministic or exploratory that behaviour looks."
            ]);
        InteractiveGenerationComparison(lesson, lesson.StartSeed, 2);

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[springgreen3]To be continued...[/]");
    }

    private static void PrintTopPredictions(InferenceLesson lesson, string seedText, int count)
    {
        AnsiConsole.MarkupLine($"[grey]Top predictions after[/] [aqua]\"{Markup.Escape(seedText)}\"[/][grey]:[/]");
        foreach (var entry in InferenceTextGenerator.TopPredictions(lesson.Model, lesson.TokenToId, lesson.IdToToken, seedText, count))
        {
            AnsiConsole.MarkupLine($"[aqua]  {Markup.Escape($"{entry.Token,-12}")}[/] [springgreen3]{entry.Probability:P1}[/]");
        }
    }

    private static void ShowStep(string title, IEnumerable<string> lines)
    {
        AnsiConsole.MarkupLine($"[bold white on red] {Markup.Escape(title)} [/]");
        AnsiConsole.MarkupLine($"[grey]{new string('=', title.Length)}[/]");
        AnsiConsole.WriteLine();

        foreach (var line in lines)
        {
            AnsiConsole.MarkupLine($"[khaki1]{Markup.Escape(line)}[/]");
        }

        AnsiConsole.WriteLine();
    }

    private static void Pause()
    {
        AnsiConsole.Markup("[grey]Press Enter to continue...[/]");
        Console.ReadLine();
        AnsiConsole.WriteLine();
    }

    private static void InteractiveGenerationComparison(InferenceLesson lesson, string startSeed, int maxTokens)
    {
        var greedyTokens = startSeed
            .Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(token => token.ToLowerInvariant())
            .ToList();

        var sampledTokens = new List<string>(greedyTokens);

        AnsiConsole.MarkupLine($"[grey]Seed:[/] [aqua]{Markup.Escape(startSeed)}[/]");
        AnsiConsole.MarkupLine($"[grey]Greedy output:[/]  [springgreen3]{Markup.Escape(string.Join(" ", greedyTokens))}[/]");
        AnsiConsole.MarkupLine($"[grey]Sampled output:[/] [springgreen3]{Markup.Escape(string.Join(" ", sampledTokens))}[/]");
        AnsiConsole.WriteLine();

        for (var step = 1; step <= maxTokens; step++)
        {
            AnsiConsole.Markup($"[grey]Press Enter to generate token {step} for both modes...[/]");
            Console.ReadLine();

            greedyTokens.Add(PredictGreedyNextToken(lesson, greedyTokens));
            sampledTokens.Add(PredictSampledNextToken(lesson, sampledTokens, temperature: 1.1, topK: 5, randomSeed: 100 + step));

            AnsiConsole.MarkupLine($"[grey]Greedy output:[/]  [springgreen3]{Markup.Escape(string.Join(" ", greedyTokens))}[/]");
            AnsiConsole.MarkupLine($"[grey]Sampled output:[/] [springgreen3]{Markup.Escape(string.Join(" ", sampledTokens))}[/]");
            AnsiConsole.WriteLine();
        }
    }

    private static string PredictGreedyNextToken(InferenceLesson lesson, IReadOnlyList<string> tokens)
    {
        var context = tokens
            .TakeLast(lesson.Model.ContextSize)
            .Select(token => lesson.TokenToId[token])
            .ToArray();

        var next = lesson.Model.PredictNext(context);
        return lesson.IdToToken[next];
    }

    private static string PredictSampledNextToken(
        InferenceLesson lesson,
        IReadOnlyList<string> tokens,
        double temperature,
        int topK,
        int randomSeed)
    {
        var context = tokens
            .TakeLast(lesson.Model.ContextSize)
            .Select(token => lesson.TokenToId[token])
            .ToArray();

        var probabilities = lesson.Model.Inspect(context).Probabilities;
        var adjusted = probabilities
            .Select((probability, index) => new
            {
                Index = index,
                Score = Math.Pow(Math.Max(probability, 1e-9), 1.0 / Math.Max(temperature, 0.05))
            })
            .OrderByDescending(entry => entry.Score)
            .Take(topK)
            .ToList();

        var random = new Random(randomSeed);
        var sum = adjusted.Sum(entry => entry.Score);
        var roll = random.NextDouble() * sum;
        var running = 0.0;

        foreach (var entry in adjusted)
        {
            running += entry.Score;
            if (roll <= running)
            {
                return lesson.IdToToken[entry.Index];
            }
        }

        return lesson.IdToToken[adjusted[^1].Index];
    }
}
