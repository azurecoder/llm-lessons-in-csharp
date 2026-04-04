namespace ukazure.llm.Lesson7;

internal static class Retriever
{
    public static List<RetrievalResult> Retrieve(string question, List<KnowledgeDocument> knowledgeBase, int top)
    {
        var queryTokens = Tokenize(question);

        return knowledgeBase
            .Select(document => new RetrievalResult
            {
                Document = document,
                Score = ScoreDocument(queryTokens, document)
            })
            .OrderByDescending(result => result.Score)
            .ThenBy(result => result.Document.Id)
            .Take(top)
            .ToList();
    }

    private static int ScoreDocument(HashSet<string> queryTokens, KnowledgeDocument document)
    {
        var documentTokens = Tokenize(document.Title + " " + document.Content);
        return queryTokens.Intersect(documentTokens).Count();
    }

    private static HashSet<string> Tokenize(string text)
    {
        return text
            .ToLowerInvariant()
            .Split([' ', '.', ',', '?', '!', ';', ':', '-', '\r', '\n', '\t'], StringSplitOptions.RemoveEmptyEntries)
            .ToHashSet();
    }
}
