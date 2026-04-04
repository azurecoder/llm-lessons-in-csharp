using Spectre.Console;

namespace ukazure.llm.Lesson1;

internal static class LessonPresenter
{
    public static void Run(TinyLlmLesson lesson)
    {
        Console.Clear();

        ShowStep(
            "1. What is a language model?",
            [
                "A language model predicts the next token.",
                "For this first demo, we make that idea tiny and concrete.",
                "There is no magic yet: just counts, lookups, and repeated prediction."
            ]);

        Pause();

        ShowStep(
            "2. Start tiny",
            [
                "We use a microscopic training corpus so every step stays visible.",
                "This lets you see exactly what the model learned from Azure-flavoured developer text."
            ]);
        AnsiConsole.MarkupLine($"[deepskyblue1]{Markup.Escape(lesson.Corpus)}[/]");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine($"[grey]Selected sentences:[/] [aqua]{lesson.SelectedSentences.Count}[/]");

        Pause();

        ShowStep(
            "3. Tokenize text",
            [
                "Tokenization means splitting text into pieces.",
                "In this demo, each word is one token."
            ]);
        AnsiConsole.MarkupLine($"[aqua]{Markup.Escape(string.Join(" | ", lesson.Tokens))}[/]");

        Pause();

        ShowStep(
            "4. Build a vocabulary",
            [
                "The vocabulary is the set of unique tokens.",
                "Each unique token gets an integer ID so the model can work with numbers."
            ]);
        foreach (var pair in lesson.TokenToId.OrderBy(pair => pair.Value))
        {
            AnsiConsole.MarkupLine($"[grey]{pair.Value,2} ->[/] [aqua]{Markup.Escape(pair.Key)}[/]");
        }

        Pause();

        ShowStep(
            "5. Create training pairs",
            [
                "Every token tries to predict the token that comes next.",
                "That turns one sequence into many tiny training examples."
            ]);
        foreach (var pair in lesson.TrainingPairs.Take(10))
        {
            AnsiConsole.MarkupLine(
                $"[aqua]{Markup.Escape($"{lesson.IdToToken[pair.Input],-5}")}[/] [grey]->[/] [springgreen3]{Markup.Escape(lesson.IdToToken[pair.Target])}[/]");
        }

        if (lesson.TrainingPairs.Count > 10)
        {
            AnsiConsole.MarkupLine($"[grey]... and[/] [aqua]{lesson.TrainingPairs.Count - 10}[/] [grey]more pairs[/]");
        }

        Pause();

        ShowStep(
            "6. Train a bigram model",
            [
                "A bigram model stores counts for what tends to follow what.",
                "It only looks one token back, so it is simple enough to inspect directly."
            ]);
        lesson.Model.PrintCounts(lesson.IdToToken);

        Pause();

        ShowStep(
            "7. Generate text",
            [
                "Generation is repeated prediction.",
                "Pick a starting token, ask for the most likely next token, and keep going.",
                "For this version, we reveal the output one token at a time so you can watch the model generate."
            ]);
        InteractiveGenerationDemo(lesson, maxTokens: 8);

        Pause();

        ShowStep(
            "8. Explain the limitations",
            [
                "This model only looks one token back.",
                "It does not understand meaning.",
                "It has no embeddings.",
                "It cannot generalize much beyond the tiny corpus."
            ]);

        Pause();

        ShowStep(
            "9. Tease the next step",
            [
                "Next we replace raw counts with learnable weights.",
                "That means introducing embeddings, a linear layer, softmax, loss, and gradient descent.",
                "Same idea, much more expressive model."
            ]);

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[springgreen3]To be continued...[/]");
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

    private static void InteractiveGenerationDemo(TinyLlmLesson lesson, int maxTokens)
    {
        var generatedTokens = new List<string> { lesson.StartToken };
        var current = lesson.TokenToId[lesson.StartToken];

        AnsiConsole.MarkupLine($"[grey]Seed:[/] [aqua]{Markup.Escape(lesson.StartToken)}[/]");
        AnsiConsole.MarkupLine($"[grey]Current output:[/] [springgreen3]{Markup.Escape(string.Join(" ", generatedTokens))}[/]");
        AnsiConsole.WriteLine();

        for (var step = 1; step <= maxTokens; step++)
        {
            AnsiConsole.Markup($"[grey]Press Enter to generate token {step}...[/]");
            Console.ReadLine();

            var next = lesson.Model.PredictNext(current);
            var nextToken = lesson.IdToToken[next];
            generatedTokens.Add(nextToken);

            AnsiConsole.MarkupLine(
                $"[grey]Step {step}:[/] [aqua]{Markup.Escape(lesson.IdToToken[current])}[/] [grey]->[/] [springgreen3]{Markup.Escape(nextToken)}[/]");
            AnsiConsole.MarkupLine($"[grey]Current output:[/] [springgreen3]{Markup.Escape(string.Join(" ", generatedTokens))}[/]");
            AnsiConsole.WriteLine();

            current = next;
        }

        AnsiConsole.MarkupLine($"[bold grey]Final output:[/] [bold springgreen3]{Markup.Escape(string.Join(" ", generatedTokens))}[/]");
    }
}
