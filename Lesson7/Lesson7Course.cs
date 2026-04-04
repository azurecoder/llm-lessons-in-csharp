using Spectre.Console;

namespace ukazure.llm.Lesson7;

internal sealed class Lesson7Course : ILesson
{
    public string Title => "Prompting, retrieval, and grounded answers";

    public void Run()
    {
        var selectedQuestion = AnsiConsole.Prompt(
            new SelectionPrompt<string>()
                .Title("Choose the [green]developer question[/] for lesson 7")
                .AddChoices(Lesson7QuestionCatalog.All));

        var lesson = RetrievalLesson.Create(selectedQuestion);
        RetrievalLessonPresenter.Run(lesson);
    }
}
