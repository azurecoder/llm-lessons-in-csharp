using Spectre.Console;

namespace ukazure.llm.Lesson3;

internal sealed class Lesson3Course : ILesson
{
    public string Title => "Model sequencing with context windows";

    public void Run()
    {
        var selectedSentences = AnsiConsole.Prompt(
            new MultiSelectionPrompt<string>()
                .Title("Choose the [green]training sentences[/] for lesson 3")
                .NotRequired()
                .InstructionsText("[grey](Press [blue]<space>[/] to toggle, [blue]<enter>[/] to confirm)[/]")
                .AddChoices(Lesson3SentenceCatalog.All));

        if (selectedSentences.Count == 0)
        {
            selectedSentences = Lesson3SentenceCatalog.All.Take(4).ToList();
        }

        var previewLesson = SequenceLesson.Create(selectedSentences);
        var startSeed = AnsiConsole.Prompt(
            new SelectionPrompt<string>()
                .Title("Choose the [green]starting context[/] for lesson 3 generation")
                .AddChoices(previewLesson.AvailableSeeds));

        var lesson = SequenceLesson.Create(selectedSentences, startSeed);
        SequenceLessonPresenter.Run(lesson);
    }
}
