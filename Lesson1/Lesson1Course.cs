using Spectre.Console;

namespace ukazure.llm.Lesson1;

internal sealed class Lesson1Course : ILesson
{
    public string Title => "A tiny bigram model built from counts";

    public void Run()
    {
        var selectedSentences = AnsiConsole.Prompt(
            new MultiSelectionPrompt<string>()
                .Title("Choose the [green]training sentences[/] for lesson 1")
                .NotRequired()
                .InstructionsText("[grey](Press [blue]<space>[/] to toggle, [blue]<enter>[/] to confirm)[/]")
                .AddChoices(Lesson1SentenceCatalog.All));

        if (selectedSentences.Count == 0)
        {
            selectedSentences = Lesson1SentenceCatalog.All.Take(4).ToList();
        }

        var previewLesson = TinyLlmLesson.Create(selectedSentences);
        var startToken = AnsiConsole.Prompt(
            new SelectionPrompt<string>()
                .Title("Choose the [green]starting token[/] for generation")
                .AddChoices(previewLesson.Vocabulary));

        var lesson = TinyLlmLesson.Create(selectedSentences, startToken);
        LessonPresenter.Run(lesson);
    }
}
