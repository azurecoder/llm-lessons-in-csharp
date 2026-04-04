using Spectre.Console;

namespace ukazure.llm.Lesson2;

internal sealed class Lesson2Course : ILesson
{
    public string Title => "Replace counts with learnable weights";

    public void Run()
    {
        var selectedSentences = AnsiConsole.Prompt(
            new MultiSelectionPrompt<string>()
                .Title("Choose the [green]training sentences[/] for lesson 2")
                .NotRequired()
                .InstructionsText("[grey](Press [blue]<space>[/] to toggle, [blue]<enter>[/] to confirm)[/]")
                .AddChoices(Lesson2SentenceCatalog.All));

        if (selectedSentences.Count == 0)
        {
            selectedSentences = Lesson2SentenceCatalog.All.Take(4).ToList();
        }

        var previewLesson = NeuralBigramLesson.Create(selectedSentences);
        var startToken = AnsiConsole.Prompt(
            new SelectionPrompt<string>()
                .Title("Choose the [green]starting token[/] for lesson 2 generation")
                .AddChoices(previewLesson.Vocabulary));

        var lesson = NeuralBigramLesson.Create(selectedSentences, startToken);
        NeuralLessonPresenter.Run(lesson);
    }
}
