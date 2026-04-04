using Spectre.Console;

namespace ukazure.llm.Lesson4;

internal sealed class Lesson4Course : ILesson
{
    public string Title => "Attention lets the model choose what to focus on";

    public void Run()
    {
        var selectedSentences = AnsiConsole.Prompt(
            new MultiSelectionPrompt<string>()
                .Title("Choose the [green]training sentences[/] for lesson 4")
                .NotRequired()
                .InstructionsText("[grey](Press [blue]<space>[/] to toggle, [blue]<enter>[/] to confirm)[/]")
                .AddChoices(Lesson4SentenceCatalog.All));

        if (selectedSentences.Count == 0)
        {
            selectedSentences = Lesson4SentenceCatalog.All.Take(4).ToList();
        }

        var previewLesson = AttentionLesson.Create(selectedSentences);
        var startSeed = AnsiConsole.Prompt(
            new SelectionPrompt<string>()
                .Title("Choose the [green]starting context[/] for lesson 4 generation")
                .AddChoices(previewLesson.AvailableSeeds));

        var lesson = AttentionLesson.Create(selectedSentences, startSeed);
        AttentionLessonPresenter.Run(lesson);
    }
}
