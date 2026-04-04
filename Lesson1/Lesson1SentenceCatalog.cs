namespace ukazure.llm.Lesson1;

internal static class Lesson1SentenceCatalog
{
    public static IReadOnlyList<string> All { get; } =
    [
        "azure deploys to the cloud",
        "azure scales in the cloud",
        "dotnet builds in the cloud",
        "dotnet runs in containers",
        "bicep deploys infra to azure",
        "github ships code to appservice",
        "python runs in containers",
        "functions scale on azure"
    ];
}
