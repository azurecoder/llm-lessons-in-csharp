using Spectre.Console;

namespace ukazure.llm.Lesson2;

internal static class NeuralLessonPresenter
{
    public static void Run(NeuralBigramLesson lesson)
    {
        Console.Clear();

        ShowStep(
            "Lesson 2: Replace counts with learnable weights",
            [
                "Last time we counted what followed what.",
                "This time we keep the same next-token prediction task, but make the model trainable."
            ]);

        Pause();

        ShowStep(
            "1. Reuse the same tiny corpus",
            [
                "Keeping the corpus fixed makes the architectural change easier to see.",
                "You can still choose the training sentences first, just like lesson 1.",
                "We are changing the model, not the data or the Azure-centric vocabulary."
            ]);
        AnsiConsole.MarkupLine($"[deepskyblue1]{Markup.Escape(lesson.Corpus)}[/]");

        Pause();

        ShowStep(
            "2. Turn tokens into IDs again",
            [
                "We still tokenize text and map each token to an integer.",
                "Neural models also need numerical inputs."
            ]);
        AnsiConsole.MarkupLine($"[aqua]{Markup.Escape(string.Join(" | ", lesson.Tokens))}[/]");
        AnsiConsole.WriteLine();
        foreach (var pair in lesson.TokenToId.OrderBy(pair => pair.Value))
        {
            AnsiConsole.MarkupLine($"[grey]{pair.Value,2} ->[/] [aqua]{Markup.Escape(pair.Key)}[/]");
        }

        Pause();

        ShowStep(
            "3. Introduce embeddings",
            [
                "Each token now gets a learnable vector instead of just a row in a count table.",
                $"In this demo the embedding size is {lesson.Model.EmbeddingSize}.",
                "Those numbers start random and improve during training."
            ]);

        Pause();

        ShowStep(
            "4. Add a linear layer",
            [
                "The embedding is passed through a linear layer to produce one score per next token.",
                "Those scores are called logits."
            ]);
        AnsiConsole.MarkupLine($"[grey]Embedding matrix shape:[/] [aqua]{lesson.Vocabulary.Count} x {lesson.Model.EmbeddingSize}[/]");
        AnsiConsole.MarkupLine($"[grey]Linear layer shape:[/] [aqua]{lesson.Model.EmbeddingSize} x {lesson.Vocabulary.Count}[/]");

        Pause();

        ShowStep(
            "5. Turn logits into probabilities with softmax",
            [
                "Softmax converts arbitrary scores into a probability distribution.",
                "Now the model can say how likely each next token is."
            ]);
        PrintTopPredictions(lesson, lesson.StartToken, 5);

        Pause();

        ShowStep(
            "6. Measure error with loss",
            [
                "We use cross-entropy loss.",
                "If the correct next token gets low probability, the loss is high.",
                "As training improves the predictions, the loss should go down."
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
            "7. Improve weights with gradient descent",
            [
                "Gradient descent nudges embeddings, linear weights, and bias values to reduce loss.",
                "That is the key difference from lesson 1: the model learns parameters instead of just storing counts."
            ]);
        AnsiConsole.MarkupLine($"[grey]Training pairs seen per epoch:[/] [aqua]{lesson.TrainingPairs.Count}[/]");
        AnsiConsole.MarkupLine("[grey]Optimiser:[/] [aqua]simple stochastic gradient descent[/]");

        Pause();

        ShowStep(
            "8. Generate text again",
            [
                "Generation is still repeated next-token prediction.",
                "But now the probabilities come from learned weights rather than raw counts.",
                "We will reveal the generated tokens step by step so you can see the model build the output."
            ]);
        InteractiveGenerationDemo(lesson, lesson.StartToken, 8);

        Pause();

        ShowStep(
            "9. Why this matters",
            [
                "Embeddings let tokens live in a shared numeric space.",
                "A linear layer plus softmax gives smooth probabilities.",
                "Loss plus gradient descent lets the model improve through optimisation."
            ]);

        Pause();

        ShowStep(
            "10. Tease the next step",
            [
                "We still only look one token back.",
                "The next leap is to use more context and let tokens interact with one another.",
                "That is where sequence models and attention start to enter the story."
            ]);

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[springgreen3]To be continued...[/]");
    }

    private static void PrintTopPredictions(NeuralBigramLesson lesson, string token, int count)
    {
        var probabilities = lesson.Model.PredictProbabilities(lesson.TokenToId[token]);
        var top = Enumerable.Range(0, probabilities.Length)
            .Select(index => new { Token = lesson.IdToToken[index], Probability = probabilities[index] })
            .OrderByDescending(entry => entry.Probability)
            .Take(count);

        AnsiConsole.MarkupLine($"[grey]Top predictions after[/] [aqua]\"{Markup.Escape(token)}\"[/][grey]:[/]");
        foreach (var entry in top)
        {
            AnsiConsole.MarkupLine($"[aqua]  {Markup.Escape($"{entry.Token,-5}")}[/] [springgreen3]{entry.Probability:P1}[/]");
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

    private static void InteractiveGenerationDemo(NeuralBigramLesson lesson, string startToken, int maxTokens)
    {
        var generatedTokens = new List<string> { startToken };
        var current = lesson.TokenToId[startToken];

        AnsiConsole.MarkupLine($"[grey]Seed:[/] [aqua]{Markup.Escape(startToken)}[/]");
        AnsiConsole.MarkupLine($"[grey]Current output:[/] [springgreen3]{Markup.Escape(string.Join(" ", generatedTokens))}[/]");
        AnsiConsole.WriteLine();

        for (var step = 1; step <= maxTokens; step++)
        {
            AnsiConsole.Markup($"[grey]Press Enter to generate token {step}...[/]");
            Console.ReadLine();

            var probabilities = lesson.Model.PredictProbabilities(current);
            var next = lesson.Model.PredictNext(current);
            var nextToken = lesson.IdToToken[next];
            generatedTokens.Add(nextToken);

            AnsiConsole.MarkupLine(
                $"[grey]Step {step}:[/] [aqua]{Markup.Escape(lesson.IdToToken[current])}[/] [grey]->[/] [springgreen3]{Markup.Escape(nextToken)}[/] [grey]({probabilities[next]:P1})[/]");
            AnsiConsole.MarkupLine($"[grey]Current output:[/] [springgreen3]{Markup.Escape(string.Join(" ", generatedTokens))}[/]");
            AnsiConsole.WriteLine();

            current = next;
        }

        AnsiConsole.MarkupLine($"[bold grey]Final output:[/] [bold springgreen3]{Markup.Escape(string.Join(" ", generatedTokens))}[/]");
    }
}
