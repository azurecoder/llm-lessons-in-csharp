namespace ukazure.llm.Lesson7;

internal static class PromptBuilder
{
    public static string BuildWithoutRetrieval(string question)
    {
        return """
            System:
            You are a helpful Azure architecture assistant.

            User:
            """ + Environment.NewLine + question;
    }

    public static string BuildWithRetrieval(string question, List<KnowledgeDocument> documents)
    {
        var context = string.Join(
            Environment.NewLine,
            documents.Select(document => $"- {document.Title}: {document.Content}"));

        return """
            System:
            You are a helpful Azure architecture assistant. Answer only using the supplied context.

            Retrieved context:
            """ + Environment.NewLine + context + Environment.NewLine + Environment.NewLine + """
            User:
            """ + Environment.NewLine + question;
    }
}
