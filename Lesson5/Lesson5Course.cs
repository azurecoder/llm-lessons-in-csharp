using Spectre.Console;

namespace ukazure.llm.Lesson5;

internal sealed class Lesson5Course : ILesson
{
    public string Title => "A tiny transformer-style block";

    public void Run()
    {
        var selectedSentences = AnsiConsole.Prompt(
            new MultiSelectionPrompt<string>()
                .Title("Choose the [green]training sentences[/] for lesson 5")
                .NotRequired()
                .InstructionsText("[grey](Press [blue]<space>[/] to toggle, [blue]<enter>[/] to confirm)[/]")
                .AddChoices(Lesson5SentenceCatalog.All));

        if (selectedSentences.Count == 0)
        {
            selectedSentences = Lesson5SentenceCatalog.All.Take(4).ToList();
        }

        var previewLesson = TransformerLesson.Create(selectedSentences);
        var startSeed = AnsiConsole.Prompt(
            new SelectionPrompt<string>()
                .Title("Choose the [green]starting context[/] for lesson 5 generation")
                .AddChoices(previewLesson.AvailableSeeds));

        var lesson = TransformerLesson.Create(selectedSentences, startSeed);
        TransformerLessonPresenter.Run(lesson);
    }
}
