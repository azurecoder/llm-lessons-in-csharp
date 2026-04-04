namespace ukazure.llm.Lesson7;

internal static class AnswerComposer
{
    public static string ComposeUngroundedAnswer(string question)
    {
        return "You should use a secure service for secrets, avoid hard-coded credentials, and prefer platform features that reduce direct secret handling.";
    }

    public static string ComposeGroundedAnswer(string question, List<KnowledgeDocument> documents)
    {
        var usesKeyVault = documents.Any(document => document.Title == "Azure Key Vault");
        var usesManagedIdentity = documents.Any(document => document.Title == "Managed Identities");
        var mentionsAppService = documents.Any(document => document.Title == "Azure App Service");

        var lines = new List<string>();

        if (usesKeyVault)
        {
            lines.Add("Store secrets in Azure Key Vault instead of appsettings files or source code.");
        }

        if (usesManagedIdentity)
        {
            lines.Add("Use managed identities so the app can authenticate without storing passwords or client secrets.");
        }

        if (mentionsAppService)
        {
            lines.Add("If the app runs on Azure App Service, configure app settings to reference Key Vault secrets.");
        }

        return string.Join(" ", lines);
    }
}
