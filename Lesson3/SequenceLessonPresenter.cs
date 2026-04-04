using Spectre.Console;

namespace ukazure.llm.Lesson3;

internal static class SequenceLessonPresenter
{
    public static void Run(SequenceLesson lesson)
    {
        Console.Clear();

        ShowStep(
            "Lesson 3: Model sequencing",
            [
                "A bigram only looks one token back.",
                "A simple neural bigram still only looks one token back.",
                "Now we let the model look at an ordered sequence of previous tokens."
            ]);

        Pause();

        ShowStep(
            "1. Why sequences matter",
            [
                "Language is not just a bag of words.",
                "Order carries meaning.",
                "The phrase before the current token changes what should come next."
            ]);

        Pause();

        ShowStep(
            "2. Reuse a tiny corpus",
            [
                "We still keep the data tiny so the mechanics stay visible.",
                "You can choose the training sentences first, just like lessons 1 and 2.",
                "But now the corpus is designed to reward using more context from Azure and programming phrases."
            ]);
        AnsiConsole.MarkupLine($"[deepskyblue1]{Markup.Escape(lesson.Corpus)}[/]");

        Pause();

        ShowStep(
            "3. Build sliding windows",
            [
                $"Instead of one previous token, each example now uses {lesson.Model.ContextSize} previous tokens.",
                "The context window slides across the corpus one token at a time."
            ]);
        foreach (var example in lesson.TrainingExamples.Take(8))
        {
            var context = string.Join(" ", example.Context.Select(id => lesson.IdToToken[id]));
            AnsiConsole.MarkupLine($"[aqua][[{Markup.Escape(context)}]][/][grey] ->[/] [springgreen3]{Markup.Escape(lesson.IdToToken[example.Target])}[/]");
        }

        if (lesson.TrainingExamples.Count > 8)
        {
            AnsiConsole.MarkupLine($"[grey]... and {lesson.TrainingExamples.Count - 8} more sequence examples[/]");
        }

        Pause();

        ShowStep(
            "4. Encode order, not just membership",
            [
                "The model sees token 1, token 2, and token 3 as different positions.",
                "That means one three-token sequence is different from another, even if they share words."
            ]);
        PrintPredictionComparison(lesson, lesson.StartSeed, FindAlternateSeed(lesson, lesson.StartSeed));

        Pause();

        ShowStep(
            "5. Sequence model architecture",
            [
                "Each token in the context gets an embedding.",
                "We concatenate those embeddings into one longer vector.",
                "A linear layer turns that whole sequence representation into next-token logits."
            ]);
        AnsiConsole.MarkupLine($"[grey]Context size:[/] [aqua]{lesson.Model.ContextSize}[/]");
        AnsiConsole.MarkupLine($"[grey]Embedding size:[/] [aqua]{lesson.Model.EmbeddingSize}[/]");
        AnsiConsole.MarkupLine($"[grey]Concatenated features:[/] [aqua]{lesson.Model.ContextSize * lesson.Model.EmbeddingSize}[/]");
        AnsiConsole.MarkupLine($"[grey]Vocabulary size:[/] [aqua]{lesson.Vocabulary.Count}[/]");

        Pause();

        ShowStep(
            "6. Train with loss and gradient descent",
            [
                "The training loop is familiar: predict, measure loss, update weights.",
                "The difference is that the prediction now depends on several ordered tokens."
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
            "7. Sequence-aware predictions",
            [
                "The same token can lead to different outcomes depending on the earlier tokens.",
                "That is the key move from token-level prediction to sequence modelling."
            ]);
        var comparisonSeeds = lesson.AvailableSeeds.Take(2).ToList();
        if (comparisonSeeds.Count == 2)
        {
            PrintPredictionComparison(lesson, comparisonSeeds[0], comparisonSeeds[1]);
        }
        else if (comparisonSeeds.Count == 1)
        {
            PrintTopPredictions(lesson, comparisonSeeds[0], 3);
        }

        Pause();

        ShowStep(
            "8. Generate with a moving context window",
            [
                "Generation still predicts one token at a time.",
                "But after each prediction, the context window shifts forward.",
                "We will reveal the generated token step by step so you can watch the context move."
            ]);
        InteractiveGenerationDemo(lesson, lesson.StartSeed, 3);

        var alternateSeed = FindAlternateSeed(lesson, lesson.StartSeed);
        if (alternateSeed is not null)
        {
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[grey]Alternate seed:[/] [aqua]{Markup.Escape(alternateSeed)}[/]");
            AnsiConsole.MarkupLine(
                $"[grey]Output:[/] [springgreen3]{Markup.Escape(SequenceTextGenerator.ContinueText(lesson.Model, lesson.TokenToId, lesson.IdToToken, alternateSeed, 1))}[/]");
        }

        Pause();

        ShowStep(
            "9. What this still cannot do",
            [
                "The context window is fixed and short.",
                "Long-range dependencies still get lost.",
                "Every position is handled by plain concatenation, not dynamic interaction."
            ]);

        Pause();

        ShowStep(
            "10. Tease the next step",
            [
                "What we really want is a model that can relate tokens to other tokens flexibly.",
                "That is the doorway to attention.",
                "Instead of a fixed window, the model can learn what to focus on."
            ]);

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[springgreen3]To be continued...[/]");
    }

    private static string? FindAlternateSeed(SequenceLesson lesson, string currentSeed)
    {
        return lesson.AvailableSeeds.FirstOrDefault(seed => !string.Equals(seed, currentSeed, StringComparison.Ordinal));
    }

    private static void PrintPredictionComparison(SequenceLesson lesson, string seedTextA, string? seedTextB)
    {
        PrintTopPredictions(lesson, seedTextA, 3);

        if (seedTextB is null)
        {
            return;
        }

        AnsiConsole.WriteLine();
        PrintTopPredictions(lesson, seedTextB, 3);
    }

    private static void PrintTopPredictions(SequenceLesson lesson, string seedText, int count)
    {
        var context = seedText
            .Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(token => lesson.TokenToId[token.ToLowerInvariant()])
            .ToArray();

        var probabilities = lesson.Model.PredictProbabilities(context);
        var top = Enumerable.Range(0, probabilities.Length)
            .Select(index => new { Token = lesson.IdToToken[index], Probability = probabilities[index] })
            .OrderByDescending(entry => entry.Probability)
            .Take(count);

        AnsiConsole.MarkupLine($"[grey]Top predictions after[/] [aqua]\"{Markup.Escape(seedText)}\"[/][grey]:[/]");
        foreach (var entry in top)
        {
            AnsiConsole.MarkupLine($"[aqua]  {Markup.Escape($"{entry.Token,-10}")}[/] [springgreen3]{entry.Probability:P1}[/]");
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

    private static void InteractiveGenerationDemo(SequenceLesson lesson, string startSeed, int maxTokens)
    {
        var generatedTokens = startSeed
            .Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(token => token.ToLowerInvariant())
            .ToList();

        AnsiConsole.MarkupLine($"[grey]Seed:[/] [aqua]{Markup.Escape(startSeed)}[/]");
        AnsiConsole.MarkupLine($"[grey]Current output:[/] [springgreen3]{Markup.Escape(string.Join(" ", generatedTokens))}[/]");
        AnsiConsole.WriteLine();

        for (var step = 1; step <= maxTokens; step++)
        {
            AnsiConsole.Markup($"[grey]Press Enter to generate token {step}...[/]");
            Console.ReadLine();

            var contextTokens = generatedTokens.TakeLast(lesson.Model.ContextSize).ToList();
            var contextIds = contextTokens.Select(token => lesson.TokenToId[token]).ToArray();
            var probabilities = lesson.Model.PredictProbabilities(contextIds);
            var next = lesson.Model.PredictNext(contextIds);
            var nextToken = lesson.IdToToken[next];
            generatedTokens.Add(nextToken);

            AnsiConsole.MarkupLine(
                $"[grey]Step {step}:[/] [aqua]{Markup.Escape(string.Join(" ", contextTokens))}[/] [grey]->[/] [springgreen3]{Markup.Escape(nextToken)}[/] [grey]({probabilities[next]:P1})[/]");
            AnsiConsole.MarkupLine($"[grey]Current output:[/] [springgreen3]{Markup.Escape(string.Join(" ", generatedTokens))}[/]");
            AnsiConsole.WriteLine();
        }

        AnsiConsole.MarkupLine($"[bold grey]Final output:[/] [bold springgreen3]{Markup.Escape(string.Join(" ", generatedTokens))}[/]");
    }
}
