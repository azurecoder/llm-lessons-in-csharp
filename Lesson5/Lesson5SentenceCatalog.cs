namespace ukazure.llm.Lesson5;

internal static class Lesson5SentenceCatalog
{
    public static IReadOnlyList<string> All { get; } =
    [
        "csharp runs on dotnet and bicep deploys to azure",
        "typescript runs on node and terraform deploys to azure",
        "python runs in containers and helm deploys to kubernetes",
        "powershell runs in terminal and azcli deploys to azure",
        "github ships code to azure and appservice serves traffic",
        "functions process queues in azure and monitor tracks logs",
        "react builds ui with fluent and api runs on dotnet",
        "bicep defines infra for azure and keyvault stores secrets"
    ];
}
