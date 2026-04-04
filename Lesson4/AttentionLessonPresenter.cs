using Spectre.Console;

namespace ukazure.llm.Lesson4;

internal static class AttentionLessonPresenter
{
    public static void Run(AttentionLesson lesson)
    {
        Console.Clear();

        ShowStep(
            "Lesson 4: Attention",
            [
                "A fixed context window is better than one-token memory.",
                "But it still treats the whole window as one block.",
                "Attention lets the model decide which earlier tokens matter most right now."
            ]);

        Pause();

        ShowStep(
            "1. The core idea",
            [
                "When predicting the next token, not every previous token is equally useful.",
                "Sometimes the model should focus on the subject.",
                "Sometimes it should focus on the object, place, or adjective."
            ]);

        Pause();

        ShowStep(
            "2. Reuse a tiny corpus",
            [
                "The corpus is still tiny and structured.",
                "You can choose the training sentences first, just like lesson 3.",
                "That makes the shifting focus easy to see in Azure and developer scenarios."
            ]);
        AnsiConsole.MarkupLine($"[deepskyblue1]{Markup.Escape(lesson.Corpus)}[/]");

        Pause();

        ShowStep(
            "3. Training examples are still context windows",
            [
                $"We use a context of {lesson.Model.ContextSize} tokens.",
                "The difference is what the model does with that window."
            ]);
        foreach (var example in lesson.TrainingExamples.Take(8))
        {
            var context = string.Join(" ", example.Context.Select(id => lesson.IdToToken[id]));
            AnsiConsole.MarkupLine($"[aqua][[{Markup.Escape(context)}]][/][grey] ->[/] [springgreen3]{Markup.Escape(lesson.IdToToken[example.Target])}[/]");
        }

        if (lesson.TrainingExamples.Count > 8)
        {
            AnsiConsole.MarkupLine($"[grey]... and {lesson.TrainingExamples.Count - 8} more attention examples[/]");
        }

        Pause();

        ShowStep(
            "4. Queries, keys, and values",
            [
                "Each token embedding is transformed into a key and a value.",
                "The current token also produces a query.",
                "The query asks: which previous keys are most relevant for this prediction?"
            ]);
        AnsiConsole.MarkupLine($"[grey]Model width:[/] [aqua]{lesson.Model.ModelWidth}[/]");
        AnsiConsole.MarkupLine($"[grey]Context size:[/] [aqua]{lesson.Model.ContextSize}[/]");

        Pause();

        ShowStep(
            "5. Attention weights are dynamic",
            [
                "The model scores each token in the context.",
                "A softmax turns those scores into attention weights.",
                "Those weights change depending on the current context."
            ]);
        PrintAttention(lesson, lesson.StartSeed);

        var alternateSeed = FindAlternateSeed(lesson, lesson.StartSeed);
        if (alternateSeed is not null)
        {
            AnsiConsole.WriteLine();
            PrintAttention(lesson, alternateSeed);
        }

        Pause();

        ShowStep(
            "6. Build a context vector from weighted values",
            [
                "The model takes a weighted average of the value vectors.",
                "Tokens with more attention contribute more to the final representation.",
                "This is the key shift: fixed window to dynamic focus inside the window."
            ]);

        Pause();

        ShowStep(
            "7. Train with loss and gradient descent",
            [
                "The model still learns by reducing next-token prediction loss.",
                "But now it can learn where to look as well as what to predict."
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
            "8. Watch the focus move",
            [
                "Different contexts produce different attention patterns.",
                "That is why attention is more flexible than plain concatenation."
            ]);
        var comparisonSeeds = lesson.AvailableSeeds.Take(2).ToList();
        if (comparisonSeeds.Count == 2)
        {
            PrintAttention(lesson, comparisonSeeds[0]);
            AnsiConsole.WriteLine();
            PrintAttention(lesson, comparisonSeeds[1]);
        }
        else if (comparisonSeeds.Count == 1)
        {
            PrintAttention(lesson, comparisonSeeds[0]);
        }

        Pause();

        ShowStep(
            "9. Generate text",
            [
                "Generation still happens one token at a time.",
                "But each prediction can focus on different parts of the recent context.",
                "We will reveal the generated tokens step by step so you can watch the focus play out."
            ]);
        InteractiveGenerationDemo(lesson, lesson.StartSeed, 2);

        if (alternateSeed is not null)
        {
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[grey]Alternate seed:[/] [aqua]{Markup.Escape(alternateSeed)}[/]");
            AnsiConsole.MarkupLine(
                $"[grey]Output:[/] [springgreen3]{Markup.Escape(AttentionTextGenerator.ContinueText(lesson.Model, lesson.TokenToId, lesson.IdToToken, alternateSeed, 2))}[/]");
        }

        Pause();

        ShowStep(
            "10. Why attention changed everything",
            [
                "Attention lets tokens interact directly.",
                "The model is no longer forced to compress all context in a rigid way first.",
                "That flexibility is a major reason transformers work so well."
            ]);

        Pause();

        ShowStep(
            "11. Tease the next step",
            [
                "Real transformers use self-attention at every position, not just the final one.",
                "They add multiple heads, feed-forward layers, residual connections, and stacking.",
                "From here, we can move from tiny attention to a transformer block."
            ]);

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[springgreen3]To be continued...[/]");
    }

    private static string? FindAlternateSeed(AttentionLesson lesson, string currentSeed)
    {
        return lesson.AvailableSeeds.FirstOrDefault(seed => !string.Equals(seed, currentSeed, StringComparison.Ordinal));
    }

    private static void PrintAttention(AttentionLesson lesson, string seedText)
    {
        var context = seedText
            .Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(token => lesson.TokenToId[token.ToLowerInvariant()])
            .ToArray();

        var inspection = lesson.Model.Inspect(context);
        var tokens = seedText.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var nextToken = lesson.IdToToken[lesson.Model.PredictNext(context)];

        AnsiConsole.MarkupLine($"[grey]Context:[/] [aqua]{Markup.Escape(seedText)}[/]");
        AnsiConsole.MarkupLine($"[grey]Predicted next token:[/] [springgreen3]{Markup.Escape(nextToken)}[/]");
        AnsiConsole.MarkupLine("[grey]Attention weights:[/]");

        for (var index = 0; index < tokens.Length; index++)
        {
            AnsiConsole.MarkupLine($"[aqua]  {Markup.Escape($"{tokens[index],-12}")}[/] [springgreen3]{inspection.Weights[index]:P1}[/]");
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

    private static void InteractiveGenerationDemo(AttentionLesson lesson, string startSeed, int maxTokens)
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
            var inspection = lesson.Model.Inspect(contextIds);
            var next = lesson.Model.PredictNext(contextIds);
            var nextToken = lesson.IdToToken[next];
            generatedTokens.Add(nextToken);

            AnsiConsole.MarkupLine(
                $"[grey]Step {step}:[/] [aqua]{Markup.Escape(string.Join(" ", contextTokens))}[/] [grey]->[/] [springgreen3]{Markup.Escape(nextToken)}[/] [grey]({inspection.Probabilities[next]:P1})[/]");
            AnsiConsole.MarkupLine($"[grey]Current output:[/] [springgreen3]{Markup.Escape(string.Join(" ", generatedTokens))}[/]");
            AnsiConsole.WriteLine();
        }

        AnsiConsole.MarkupLine($"[bold grey]Final output:[/] [bold springgreen3]{Markup.Escape(string.Join(" ", generatedTokens))}[/]");
    }
}
