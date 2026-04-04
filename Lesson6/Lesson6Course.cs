using Spectre.Console;

namespace ukazure.llm.Lesson6;

internal sealed class Lesson6Course : ILesson
{
    public string Title => "Training, inference, and sampling";

    public void Run()
    {
        var selectedSentences = AnsiConsole.Prompt(
            new MultiSelectionPrompt<string>()
                .Title("Choose the [green]training sentences[/] for lesson 6")
                .NotRequired()
                .InstructionsText("[grey](Press [blue]<space>[/] to toggle, [blue]<enter>[/] to confirm)[/]")
                .AddChoices(Lesson6SentenceCatalog.All));

        if (selectedSentences.Count == 0)
        {
            selectedSentences = Lesson6SentenceCatalog.All.Take(4).ToList();
        }

        var previewLesson = InferenceLesson.Create(selectedSentences);
        var startSeed = AnsiConsole.Prompt(
            new SelectionPrompt<string>()
                .Title("Choose the [green]starting context[/] for lesson 6 generation")
                .AddChoices(previewLesson.AvailableSeeds));

        var lesson = InferenceLesson.Create(selectedSentences, startSeed);
        InferenceLessonPresenter.Run(lesson);
    }
}
