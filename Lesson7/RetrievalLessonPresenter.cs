using Spectre.Console;

namespace ukazure.llm.Lesson7;

internal static class RetrievalLessonPresenter
{
    public static void Run(RetrievalLesson lesson)
    {
        Console.Clear();

        ShowStep(
            "Lesson 7: Prompting, retrieval, and grounded answers",
            [
                "We have explained how the model works internally.",
                "Now we switch to application architecture: how developers make the model useful in a real system.",
                "This lesson is about request construction, retrieval, and grounding."
            ]);

        Pause();

        ShowStep(
            "1. The model only sees what is in the prompt",
            [
                "At inference time the model cannot magically query your systems.",
                "It only sees the tokens you send in the request.",
                "That makes prompt construction part of the application architecture."
            ]);

        Pause();

        ShowStep(
            "2. Start with a developer question",
            [
                "We will use one Azure developer question throughout the lesson.",
                "You can choose the question before the lesson starts, just like the earlier interactive lessons."
            ]);
        AnsiConsole.MarkupLine($"[grey]Question:[/] [aqua]{Markup.Escape(lesson.UserQuestion)}[/]");

        Pause();

        ShowStep(
            "3. Prompting is structured input design",
            [
                "A better prompt gives the model clearer constraints.",
                "This is closer to API design than to magic incantation."
            ]);
        AnsiConsole.MarkupLine($"[deepskyblue1]{Markup.Escape(lesson.PromptWithoutRetrieval)}[/]");

        Pause();

        ShowStep(
            "4. Add a retrieval layer",
            [
                "Before calling the model, the application can fetch relevant documents.",
                "That is the core RAG pattern: retrieve, then augment the prompt."
            ]);
        foreach (var result in lesson.RetrievedDocuments)
        {
            AnsiConsole.MarkupLine(
                $"[aqua]{Markup.Escape($"{result.Document.Title,-24}")}[/] [grey]score={result.Score} tag={Markup.Escape(result.Document.Tag)}[/]");
        }

        Pause();

        ShowStep(
            "5. Build a grounded prompt",
            [
                "Now we inject the retrieved context into the request.",
                "The model is no longer answering from generic patterns alone."
            ]);
        AnsiConsole.MarkupLine($"[deepskyblue1]{Markup.Escape(lesson.PromptWithRetrieval)}[/]");

        Pause();

        ShowStep(
            "6. Compare answers",
            [
                "Without retrieval, the answer is generic.",
                "With retrieval, the answer can be tied to specific Azure services and practices."
            ]);
        AnsiConsole.MarkupLine("[grey]Without retrieval:[/]");
        AnsiConsole.MarkupLine($"[khaki1]{Markup.Escape(lesson.UngroundedAnswer)}[/]");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[grey]With retrieval:[/]");
        AnsiConsole.MarkupLine($"[springgreen3]{Markup.Escape(lesson.GroundedAnswer)}[/]");

        Pause();

        ShowStep(
            "7. Call a live LLM through the OpenAI SDK",
            [
                "This version of the lesson can now send the grounded prompt to a real model.",
                "If no API key is configured, the lesson falls back to the local grounded answer so the demo still runs."
            ]);
        var llmResult = OpenAiGroundedResponder.GenerateGroundedAnswer(lesson);
        AnsiConsole.MarkupLine($"[grey]{Markup.Escape(llmResult.Explanation)}[/]");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[grey]LLM answer:[/]");
        AnsiConsole.MarkupLine($"[springgreen3]{Markup.Escape(llmResult.Answer)}[/]");

        Pause();

        ShowStep(
            "8. Why this matters to software developers",
            [
                "You usually do not need to retrain the model to make it useful.",
                "Very often you need to improve the request pipeline around the model."
            ]);
        AnsiConsole.MarkupLine("[grey]Developer translation:[/]");
        AnsiConsole.MarkupLine("[aqua]- prompt = request payload[/]");
        AnsiConsole.MarkupLine("[aqua]- retrieval = middleware that augments the request[/]");
        AnsiConsole.MarkupLine("[aqua]- grounding = constraining the answer with application-provided context[/]");

        Pause();

        ShowStep(
            "9. Where embeddings come back in",
            [
                "Earlier, embeddings lived inside the model.",
                "In production systems, embeddings can also live outside the model to power document retrieval."
            ]);
        AnsiConsole.MarkupLine("[grey]Same word, two roles:[/]");
        AnsiConsole.MarkupLine("[aqua]- model embeddings help the model reason[/]");
        AnsiConsole.MarkupLine("[aqua]- retrieval embeddings help your app find relevant content[/]");

        Pause();

        ShowStep(
            "10. Azure architecture connection",
            [
                "This is where services like Azure AI Search start to fit into the story.",
                "The app retrieves relevant material, builds the prompt, calls the model, and returns a grounded answer."
            ]);
        InteractiveRequestFlow(lesson);

        Pause();

        ShowStep(
            "11. The key mental model",
            [
                "Prompting changes the request.",
                "Retrieval changes the context.",
                "Grounding changes how trustworthy the answer can be."
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

    private static void InteractiveRequestFlow(RetrievalLesson lesson)
    {
        AnsiConsole.MarkupLine($"[grey]Question:[/] [aqua]{Markup.Escape(lesson.UserQuestion)}[/]");
        AnsiConsole.MarkupLine("[grey]Request flow:[/]");
        AnsiConsole.WriteLine();

        AnsiConsole.Markup("[grey]Press Enter to retrieve documents...[/]");
        Console.ReadLine();
        foreach (var result in lesson.RetrievedDocuments)
        {
            AnsiConsole.MarkupLine($"[aqua]Retrieved:[/] [springgreen3]{Markup.Escape(result.Document.Title)}[/] [grey](score {result.Score})[/]");
        }

        AnsiConsole.WriteLine();
        AnsiConsole.Markup("[grey]Press Enter to build the grounded prompt...[/]");
        Console.ReadLine();
        AnsiConsole.MarkupLine("[aqua]Prompt built with retrieved context[/]");

        AnsiConsole.WriteLine();
        AnsiConsole.Markup("[grey]Press Enter to produce the grounded answer...[/]");
        Console.ReadLine();
        AnsiConsole.MarkupLine($"[springgreen3]{Markup.Escape(lesson.GroundedAnswer)}[/]");
    }
}
