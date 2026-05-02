using Spectre.Console;

namespace ukazure.llm.NanoGpt;

internal static class NanoGptPresenter
{
    public static void Run(NanoGptRun run)
    {
        Console.Clear();

        ShowStep(
            "nanoGPT: C# with TorchSharp",
            [
                "This is no longer lesson 8. It is the practical nanoGPT track.",
                "The hand-rolled teaching arrays are replaced with TorchSharp tensors, autograd, cross-entropy, and AdamW.",
                "We train against a document, not a few hardcoded sentences."
            ]);

        Pause();

        ShowStep(
            "1. Training document",
            [
                "nanoGPT treats text as a long stream of token IDs.",
                "For this C# version we start with a small Azure-flavoured document so the output stays recognisable."
            ]);
        AnsiConsole.MarkupLine($"[grey]Path:[/] [aqua]{Markup.Escape(run.TrainingDocumentPath)}[/]");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine($"[deepskyblue1]{Markup.Escape(run.TrainingDocumentPreview)}[/]");

        Pause();

        ShowStep(
            "2. Character tokenizer",
            [
                "The first version stays character-level, like nanoGPT's quickest Shakespeare path.",
                "That keeps the vocabulary tiny and makes the training data easy to inspect."
            ]);
        AnsiConsole.MarkupLine($"[grey]Vocabulary size:[/] [aqua]{run.Tokenizer.VocabularySize}[/]");
        AnsiConsole.MarkupLine($"[grey]Vocabulary preview:[/] [aqua]{Markup.Escape(new string(run.Tokenizer.Vocabulary.Take(40).ToArray()))}[/]");

        Pause();

        ShowStep(
            "3. TorchSharp configuration",
            [
                "The model uses fixed-length character blocks.",
                "TorchSharp handles tensors, gradients, and AdamW so the code is much closer to a practical C# training loop."
            ]);
        AnsiConsole.MarkupLine($"[grey]Block size:[/]     [aqua]{run.Config.BlockSize}[/]");
        AnsiConsole.MarkupLine($"[grey]Embedding size:[/] [aqua]{run.Config.EmbeddingSize}[/]");
        AnsiConsole.MarkupLine($"[grey]Hidden size:[/]    [aqua]{run.Config.FeedForwardSize}[/]");

        Pause();

        ShowStep(
            "4. Train with AdamW",
            [
                "Each step samples batches from the document.",
                "The model predicts the next character after each block.",
                "Cross-entropy measures the mistake and AdamW updates the weights."
            ]);
        foreach (var checkpoint in run.Checkpoints)
        {
            AnsiConsole.MarkupLine($"[grey]Step {checkpoint.Step,3}:[/] [springgreen3]loss = {checkpoint.Loss:F4}[/]");
        }

        Pause();

        ShowStep(
            "5. Sample from the trained model",
            [
                "Generation now uses the trained TorchSharp model.",
                "It still appends one character at a time, but the probabilities come from tensors and learned weights."
            ]);
        AnsiConsole.MarkupLine($"[grey]Prompt:[/] [aqua]{Markup.Escape(run.Prompt)}[/]");
        AnsiConsole.MarkupLine($"[grey]Sample:[/] [springgreen3]{Markup.Escape(run.Sample.ReplaceLineEndings("\\n"))}[/]");

        Pause();

        ShowStep(
            "6. Ask the training document",
            [
                "Now you can ask questions against the same document we trained from.",
                "The tiny character model learns next-character patterns, not meaning, so the question loop uses grounded document lookup for useful answers.",
                "This keeps the demo honest: generation shows the trained model, questions show retrieval over the source text."
            ]);
        RunQuestionLoop(run);

        Pause();

        ShowStep(
            "7. Why this matters",
            [
                "For a practical C# nanoGPT, TorchSharp gives us tensors, autograd, AdamW, and a much shorter life crisis.",
                "The remaining work is to replace this compact starter model with the full nanoGPT block stack.",
                "That means causal self-attention, layer norm, multi-head attention, and checkpoint save/load."
            ]);

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[springgreen3]To be continued...[/]");
    }

    private static void RunQuestionLoop(NanoGptRun run)
    {
        AnsiConsole.MarkupLine("[grey]Ask questions about[/] [aqua]data/nanogpt-training.txt[/][grey]. Press Enter on a blank question to finish.[/]");
        AnsiConsole.MarkupLine("[grey]Try:[/] [aqua]What stores secrets?[/] [grey]or[/] [aqua]What does TorchSharp give C#?[/]");
        AnsiConsole.WriteLine();

        while (true)
        {
            AnsiConsole.Markup("[grey]Question:[/] ");
            Console.ForegroundColor = ConsoleColor.Cyan;
            var question = Console.ReadLine();
            Console.ResetColor();

            if (string.IsNullOrWhiteSpace(question))
            {
                AnsiConsole.WriteLine();
                return;
            }

            var answer = DocumentQuestionAnswerer.Ask(run.TrainingDocument, question);

            AnsiConsole.MarkupLine($"[grey]Answer:[/] [springgreen3]{Markup.Escape(answer.Answer)}[/]");

            if (answer.Evidence.Count > 0)
            {
                AnsiConsole.MarkupLine("[grey]Evidence:[/]");
                foreach (var (text, score) in answer.Evidence)
                {
                    AnsiConsole.MarkupLine($"[aqua]  {Markup.Escape(text)}[/] [grey](score {score})[/]");
                }
            }

            AnsiConsole.WriteLine();
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
}
