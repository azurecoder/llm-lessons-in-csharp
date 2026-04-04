namespace ukazure.llm.Lesson4;

internal static class Lesson4SentenceCatalog
{
    public static IReadOnlyList<string> All { get; } =
    [
        "azure stores secrets in keyvault service",
        "azure stores logs in monitor workspace",
        "dotnet builds api with minimal hosting",
        "react builds ui with fluent components",
        "bicep deploys apps to azure subscription",
        "github ships code to azure appservice",
        "python reads config from appsettings json",
        "functions process queues in azure storage"
    ];
}
