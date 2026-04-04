using Spectre.Console;

namespace ukazure.llm.Lesson5;

internal static class TransformerLessonPresenter
{
    public static void Run(TransformerLesson lesson)
    {
        Console.Clear();

        ShowStep(
            "Lesson 5: A tiny transformer-style block",
            [
                "Now we assemble the pieces into one compact block.",
                "This is still simplified, but it looks much more like the architecture people mean by 'transformer'.",
                "We combine token embeddings, positional information, self-attention, residual paths, and a feed-forward layer."
            ]);

        Pause();

        ShowStep(
            "1. Why another step after attention?",
            [
                "Attention was the key mechanism.",
                "A transformer block turns that mechanism into a reusable building unit.",
                "Real models stack many of these blocks on top of one another."
            ]);

        Pause();

        ShowStep(
            "2. The tiny training corpus",
            [
                "The corpus is still small enough to inspect.",
                "You can choose the training sentences first, just like lesson 4.",
                "But it has enough repetition to reward relational pattern matching in Azure and programming text."
            ]);
        AnsiConsole.MarkupLine($"[deepskyblue1]{Markup.Escape(lesson.Corpus)}[/]");

        Pause();

        ShowStep(
            "3. Positional information matters",
            [
                "Transformers do not get order for free.",
                "So we add positional embeddings to token embeddings.",
                "That tells the model whether a token is first, second, third, or fourth in the context."
            ]);
        AnsiConsole.MarkupLine($"[grey]Context size:[/] [aqua]{lesson.Model.ContextSize}[/]");
        AnsiConsole.MarkupLine($"[grey]Model width:[/] [aqua]{lesson.Model.ModelWidth}[/]");

        Pause();

        ShowStep(
            "4. Self-attention inside the block",
            [
                "The final token position forms a query.",
                "All positions contribute keys and values.",
                "Attention decides which earlier tokens matter for the next prediction."
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
            "5. Residual connection",
            [
                "After attention, we add the attention output back to the original representation.",
                "That skip path is called a residual connection.",
                "It helps the model preserve useful information while still learning a correction."
            ]);
        AnsiConsole.MarkupLine("[grey]Representation[/] [aqua]+[/] [grey]attention output[/] [aqua]->[/] [springgreen3]richer representation[/]");

        Pause();

        ShowStep(
            "6. Feed-forward layer",
            [
                "Then each position goes through a small neural network.",
                "In this demo that network is linear to ReLU to linear.",
                "That gives the block extra nonlinear capacity beyond attention alone."
            ]);
        AnsiConsole.MarkupLine($"[grey]Feed-forward hidden width:[/] [aqua]{lesson.Model.HiddenWidth}[/]");

        Pause();

        ShowStep(
            "7. Another residual path and output head",
            [
                "The feed-forward output is added back in through another residual path.",
                "Then a final linear layer turns the representation into logits over the vocabulary.",
                "This tiny lesson omits layer normalisation to keep the code readable."
            ]);

        Pause();

        ShowStep(
            "8. Train the block",
            [
                "The training loop is still the same familiar story.",
                "Predict the next token, measure loss, update weights."
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
            "9. Watch the block focus",
            [
                "Different contexts still produce different attention patterns.",
                "But now those attention outputs are flowing through a more transformer-like block."
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
            "10. Generate text",
            [
                "We can now generate with the tiny block in the same next-token way as before.",
                "The difference is that each prediction comes from a more expressive computation graph.",
                "We will reveal the generated tokens step by step so you can watch the block build the output."
            ]);
        InteractiveGenerationDemo(lesson, lesson.StartSeed, 2);

        if (alternateSeed is not null)
        {
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[grey]Alternate seed:[/] [aqua]{Markup.Escape(alternateSeed)}[/]");
            AnsiConsole.MarkupLine(
                $"[grey]Output:[/] [springgreen3]{Markup.Escape(TransformerTextGenerator.ContinueText(lesson.Model, lesson.TokenToId, lesson.IdToToken, alternateSeed, 2))}[/]");
        }

        Pause();

        ShowStep(
            "11. What real transformers add",
            [
                "Real transformer blocks apply self-attention at every position, not only the final one.",
                "They also use multi-head attention, layer normalisation, deeper stacks, and larger feed-forward networks.",
                "But the essential shape is already here."
            ]);

        Pause();

        ShowStep(
            "12. Where the story can go next",
            [
                "From here you can talk about stacking many blocks, training at scale, and pretraining.",
                "Or you can pivot to inference, prompting, fine-tuning, and agentic behaviour.",
                "At this point you have the conceptual ladder from counts to transformer blocks."
            ]);

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[springgreen3]To be continued...[/]");
    }

    private static string? FindAlternateSeed(TransformerLesson lesson, string currentSeed)
    {
        return lesson.AvailableSeeds.FirstOrDefault(seed => !string.Equals(seed, currentSeed, StringComparison.Ordinal));
    }

    private static void PrintAttention(TransformerLesson lesson, string seedText)
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
            AnsiConsole.MarkupLine($"[aqua]  {Markup.Escape($"{tokens[index],-12}")}[/] [springgreen3]{inspection.AttentionWeights[index]:P1}[/]");
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

    private static void InteractiveGenerationDemo(TransformerLesson lesson, string startSeed, int maxTokens)
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
