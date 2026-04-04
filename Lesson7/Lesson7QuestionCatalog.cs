namespace ukazure.llm.Lesson7;

internal static class Lesson7QuestionCatalog
{
    public static IReadOnlyList<string> All { get; } =
    [
        "How should I store secrets for my Azure app without hard-coding credentials?",
        "How should I host a .NET API on Azure without managing servers?",
        "How can I let my Azure app call other Azure services without storing passwords?",
        "What Azure service should I use for containerised apps with managed scaling?"
    ];
}
