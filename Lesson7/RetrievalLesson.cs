namespace ukazure.llm.Lesson7;

internal sealed class RetrievalLesson
{
    public required string UserQuestion { get; init; }
    public required List<KnowledgeDocument> KnowledgeBase { get; init; }
    public required List<RetrievalResult> RetrievedDocuments { get; init; }
    public required string PromptWithoutRetrieval { get; init; }
    public required string PromptWithRetrieval { get; init; }
    public required string UngroundedAnswer { get; init; }
    public required string GroundedAnswer { get; init; }
    public required string SuggestedModel { get; init; }

    public static RetrievalLesson Create(string userQuestion)
    {
        var knowledgeBase = new List<KnowledgeDocument>
        {
            new()
            {
                Id = "doc1",
                Title = "Azure Key Vault",
                Content = "Azure Key Vault stores secrets, keys, and certificates. Applications can use managed identities to access Key Vault without storing credentials in code or configuration.",
                Tag = "security"
            },
            new()
            {
                Id = "doc2",
                Title = "Managed Identities",
                Content = "Managed identities let Azure resources authenticate to other Azure services without developers managing passwords, client secrets, or connection strings.",
                Tag = "identity"
            },
            new()
            {
                Id = "doc3",
                Title = "Azure App Service",
                Content = "Azure App Service is a managed hosting platform for web apps and APIs. App settings can reference secrets from Azure Key Vault.",
                Tag = "hosting"
            },
            new()
            {
                Id = "doc4",
                Title = "Azure Container Apps",
                Content = "Azure Container Apps runs containerized applications and supports secrets for application configuration. It can integrate with managed identities for secure access to dependencies.",
                Tag = "containers"
            },
            new()
            {
                Id = "doc5",
                Title = "Azure Functions",
                Content = "Azure Functions lets developers run event-driven code without managing servers. It supports managed identities and integrates with Key Vault and other Azure services.",
                Tag = "serverless"
            },
            new()
            {
                Id = "doc6",
                Title = "Azure AI Search",
                Content = "Azure AI Search can index enterprise content and retrieve relevant chunks for retrieval-augmented generation workflows. It is commonly used to ground model responses with business data.",
                Tag = "search"
            }
        };

        var retrievedDocuments = Retriever.Retrieve(userQuestion, knowledgeBase, 3);
        var promptWithoutRetrieval = PromptBuilder.BuildWithoutRetrieval(userQuestion);
        var promptWithRetrieval = PromptBuilder.BuildWithRetrieval(userQuestion, retrievedDocuments.Select(result => result.Document).ToList());
        var ungroundedAnswer = AnswerComposer.ComposeUngroundedAnswer(userQuestion);
        var groundedAnswer = AnswerComposer.ComposeGroundedAnswer(userQuestion, retrievedDocuments.Select(result => result.Document).ToList());

        return new RetrievalLesson
        {
            UserQuestion = userQuestion,
            KnowledgeBase = knowledgeBase,
            RetrievedDocuments = retrievedDocuments,
            PromptWithoutRetrieval = promptWithoutRetrieval,
            PromptWithRetrieval = promptWithRetrieval,
            UngroundedAnswer = ungroundedAnswer,
            GroundedAnswer = groundedAnswer,
            SuggestedModel = "gpt-5-mini"
        };
    }
}
